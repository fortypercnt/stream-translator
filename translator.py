import argparse
import sys
from datetime import datetime

import ffmpeg
import numpy as np
import whisper

SAMPLE_RATE = 16000


def open_stream(stream, directURL):
    if not directURL:
        import streamlink
        stream_options = streamlink.streams(stream)
        if not stream_options:
            print("No playable streams found on this URL:", stream)
            sys.exit(0)

        if 'audio_only' in stream_options:
            stream = stream_options['audio_only'].url
        elif 'best' in stream_options:
            stream = stream_options['best'].url
        else:
            stream = next(iter(stream_options.values())).url

    try:
        process = (
            ffmpeg.input(stream, loglevel="panic")
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run_async(pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return process


def main(args):
    url = args.URL
    model = args.model
    task = args.task
    language = args.language
    interval = args.interval
    beam_size = args.beam_size
    best_of = args.best_of
    direct_url = args.directURL

    n_samples = interval * SAMPLE_RATE * 2  # Factor 2 comes from reading the int16 stream as bytes
    if beam_size <= 0 or best_of <= 0:
        beam_size = None
        best_of = None

    if model.endswith('.en'):
        if model == 'large.en':
            print("English model does not have large model, please choose from {tiny.en, small.en, medium.en}")
            sys.exit(0)
        if language != 'English' and language != 'en':
            if language == 'auto':
                print("Using .en model, setting language from auto to English")
                language = 'en'
            else:
                print("English model cannot be used to detect non english language, please choose a non .en model")
                sys.exit(0)
    lang_str = ""
    if language == 'auto':
        language = None

    print("Loading model...")
    model = whisper.load_model(model)

    print("Opening stream...")
    process = open_stream(url, direct_url)

    while process.poll() is None:
        try:
            # Load audio from ffmpeg stream
            in_bytes = process.stdout.read(n_samples)
            if not in_bytes:
                break
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0

            # Decode the audio
            result = model.transcribe(audio, task=task, language=language, beam_size=beam_size, best_of=best_of,
                                      compression_ratio_threshold=2.0, suppress_blank=False, without_timestamps=True)
            if language is None:
                lang_str = "(" + result.get("language") + ")"
            print(f'{datetime.now().strftime("%H:%M:%S")} {lang_str} {result.get("text")}')

        except Exception as e:
            print("Error", e)

    print("Stream ended")
    process.wait()


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
                        help='Interval between calls to the language model in seconds. Should not be higher than 30.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Number of beams in beam search. Set to 0 to use greedy algorithm instead.')
    parser.add_argument('--best_of', type=int, default=5,
                        help='Number of candidates when sampling with non-zero temperature.')
    parser.add_argument('--directURL', action='store_true',
                        help='Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to '
                             'obtain the stream URL.')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
