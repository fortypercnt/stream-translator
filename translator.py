import argparse
import sys
from datetime import datetime

import ffmpeg
import numpy as np
import whisper
from whisper.audio import SAMPLE_RATE


class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.data = []
        self.full = False
        self.cur = 0

    def append(self, x):
        if self.size <= 0:
            return
        if self.full:
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.size
        else:
            self.data.append(x)
            if len(self.data) == self.size:
                self.full = True

    def get_all(self):
        """ Get all elements in chronological order from oldest to newest. """
        all_data = []
        for i in range(len(self.data)):
            idx = (i + self.cur) % self.size
            all_data.append(self.data[idx])
        return all_data

    def has_repetition(self):
        prev = None
        for elem in self.data:
            if elem == prev:
                return True
            prev = elem
        return False

    def clear(self):
        self.data = []
        self.full = False
        self.cur = 0


def open_stream(stream, direct_url, preferred_quality):
    if direct_url:
        try:
            process = (
                ffmpeg.input(stream, loglevel="panic")
                .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
                .run_async(pipe_stdout=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return process, None

    import streamlink
    import subprocess
    import threading
    stream_options = streamlink.streams(stream)
    if not stream_options:
        print("No playable streams found on this URL:", stream)
        sys.exit(0)

    option = None
    for quality in [preferred_quality, 'audio_only', 'audio_mp4a', 'audio_opus', 'best']:
        if quality in stream_options:
            option = quality
            break
    if option is None:
        # Fallback
        option = next(iter(stream_options.values()))

    def writer(streamlink_proc, ffmpeg_proc):
        while (not streamlink_proc.poll()) and (not ffmpeg_proc.poll()):
            try:
                chunk = streamlink_proc.stdout.read(1024)
                ffmpeg_proc.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                pass

    cmd = ['streamlink', stream, option, "-O"]
    streamlink_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    try:
        ffmpeg_process = (
            ffmpeg.input("pipe:", loglevel="panic")
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    thread = threading.Thread(target=writer, args=(streamlink_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, streamlink_process


def main(url, model="small", language=None, interval=5, history_buffer_size=0, preferred_quality="audio_only",
         use_vad=True, direct_url=False, faster_whisper_args=None, **decode_options):

    n_bytes = interval * SAMPLE_RATE * 2  # Factor 2 comes from reading the int16 stream as bytes
    audio_buffer = RingBuffer((history_buffer_size // interval) + 1)
    previous_text = RingBuffer(history_buffer_size // interval)

    print("Loading model...")
    if faster_whisper_args:
        from faster_whisper import WhisperModel
        model = WhisperModel(faster_whisper_args["model_path"],
                             device=faster_whisper_args["device"], 
                             compute_type=faster_whisper_args["compute_type"])
    else:
        model = whisper.load_model(model)
        
    if use_vad:
        from vad import VAD
        vad = VAD()

    print("Opening stream...")
    ffmpeg_process, streamlink_process = open_stream(url, direct_url, preferred_quality)
    try:
        while ffmpeg_process.poll() is None:
            # Read audio from ffmpeg stream
            in_bytes = ffmpeg_process.stdout.read(n_bytes)
            if not in_bytes:
                break

            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            if use_vad and vad.no_speech(audio):
                print(f'{datetime.now().strftime("%H:%M:%S")}')
                continue
            audio_buffer.append(audio)

            # Decode the audio
            clear_buffers = False
            if faster_whisper_args:
                segments, info = model.transcribe(audio,
                                                  language=language,
                                                  **decode_options)
                
                decoded_language = "" if language else "(" + info.language + ")"
                decoded_text = ""
                previous_segment = ""
                for segment in segments:
                    if segment.text != previous_segment:
                        decoded_text += segment.text
                        previous_segment = segment.text
                        
                new_prefix = decoded_text
    
            else:
                result = model.transcribe(np.concatenate(audio_buffer.get_all()),
                                          prefix="".join(previous_text.get_all()),
                                          language=language,
                                          without_timestamps=True,
                                          **decode_options)
                
                decoded_language = "" if language else "(" + result.get("language") + ")"
                decoded_text = result.get("text")
                new_prefix = ""
                for segment in result["segments"]:
                    if segment["temperature"] < 0.5 and segment["no_speech_prob"] < 0.6:
                        new_prefix += segment["text"]
                    else:
                        # Clear history if the translation is unreliable, otherwise prompting on this leads to
                        # repetition and getting stuck.
                        clear_buffers = True

            previous_text.append(new_prefix)

            if clear_buffers or previous_text.has_repetition():
                audio_buffer.clear()
                previous_text.clear()
                
            print(f'{datetime.now().strftime("%H:%M:%S")} {decoded_language} {decoded_text}')

        print("Stream ended")
    finally:
        ffmpeg_process.kill()
        if streamlink_process:
            streamlink_process.kill()


def cli():
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('URL', type=str, help='Stream website and channel name, e.g. twitch.tv/forsen')
    parser.add_argument('--model', type=str,
                        choices=['tiny', 'tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
                        default='small',
                        help='Model to be used for generating audio transcription. Smaller models are faster and use '
                             'less VRAM, but are also less accurate. .en models are more accurate but only work on '
                             'English audio.')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='translate',
                        help='Whether to transcribe the audio (keep original language) or translate to English.')
    parser.add_argument('--language', type=str, default='auto',
                        help='Language spoken in the stream. Default option is to auto detect the spoken language. '
                             'See https://github.com/openai/whisper for available languages.')
    parser.add_argument('--interval', type=int, default=5,
                        help='Interval between calls to the language model in seconds.')
    parser.add_argument('--history_buffer_size', type=int, default=0,
                        help='Seconds of previous audio/text to use for conditioning the model. Set to 0 to just use '
                             'audio from the last interval. Note that this can easily lead to repetition/loops if the'
                             'chosen language/model settings do not produce good results to begin with.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams in beam search. Set to 0 to use greedy algorithm instead.')
    parser.add_argument('--best_of', type=int, default=5,
                        help='Number of candidates when sampling with non-zero temperature.')
    parser.add_argument('--preferred_quality', type=str, default='audio_only',
                        help='Preferred stream quality option. "best" and "worst" should always be available. Type '
                             '"streamlink URL" in the console to see quality options for your URL.')
    parser.add_argument('--disable_vad', action='store_true',
                        help='Set this flag to disable additional voice activity detection by Silero VAD.')
    parser.add_argument('--direct_url', action='store_true',
                        help='Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to '
                             'obtain the stream URL.')
    parser.add_argument('--use_faster_whisper', action='store_true',
                        help='Set this flag to use faster-whisper implementation instead of the original OpenAI '
                             'implementation.')
    parser.add_argument('--faster_whisper_model_path', type=str, default='whisper-large-v2-ct2/',
                        help='Path to a directory containing a Whisper model in the CTranslate2 format.')
    parser.add_argument('--faster_whisper_device', type=str, choices=['cuda', 'cpu', 'auto'], default='cuda',
                        help='Set the device to run faster-whisper on.')
    parser.add_argument('--faster_whisper_compute_type', type=str, choices=['int8', 'int8_float16', 'int16', 'float16'],
                        default='float16',
                        help='Set the quantization type for faster-whisper. See '
                             'https://opennmt.net/CTranslate2/quantization.html for more info.')

    args = parser.parse_args().__dict__
    url = args.pop("URL")
    args["use_vad"] = not args.pop("disable_vad")
    use_faster_whisper = args.pop("use_faster_whisper")
    faster_whisper_args = dict()
    faster_whisper_args["model_path"] = args.pop("faster_whisper_model_path")
    faster_whisper_args["device"] = args.pop("faster_whisper_device")
    faster_whisper_args["compute_type"] = args.pop("faster_whisper_compute_type")

    if args['model'].endswith('.en'):
        if args['model'] == 'large.en':
            print("English model does not have large model, please choose from {tiny.en, small.en, medium.en}")
            sys.exit(0)
        if args['language'] != 'English' and args['language'] != 'en':
            if args['language'] == 'auto':
                print("Using .en model, setting language from auto to English")
                args['language'] = 'en'
            else:
                print("English model cannot be used to detect non english language, please choose a non .en model")
                sys.exit(0)

    if args['language'] == 'auto':
        args['language'] = None

    if args['beam_size'] == 0:
        args['beam_size'] = None

    main(url, faster_whisper_args=faster_whisper_args if use_faster_whisper else None, **args)


if __name__ == '__main__':
    cli()
