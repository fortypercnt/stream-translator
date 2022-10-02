import sys, argparse
from datetime import datetime
import numpy as np
import ffmpeg
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
            .global_args("-re") # Argument to act as a live stream
            .run_async(pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        
    return process
    
def main(args):
    URL: str=args.URL
    model: str=args.model
    task: str=args.task
    language_setting: str=args.language
    interval: int=args.interval
    directURL: bool=args.directURL
    
    N_SAMPLES = interval * SAMPLE_RATE * 2
    
    if model.endswith('.en'):
        if model == 'large.en':
            print("English model does not have large model, please choose from {tiny.en, small.en, medium.en}")
            sys.exit(0)
        if language_setting != 'English' and language_setting != 'en':
            if language_setting == 'auto':
                print("Using .en model, setting language from auto to English")
                language_setting = 'en'
            else:
                print("English model cannot be used to detect non english language, please choose a non .en model")
                sys.exit(0)
    language = language_setting
    lang_str = ""

    print("Loading model...")
    model = whisper.load_model(model)
    
    print("Opening stream...")
    process = open_stream(URL, directURL)
    
    while process.poll() is None:
        try:
            # Load audio and pad/trim it to fit 30 seconds
            in_bytes = process.stdout.read(N_SAMPLES)
            if not in_bytes: break
            audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            
            if language_setting == "auto":
                # Detect the spoken language
                _, probs = model.detect_language(mel)
                language = max(probs, key=probs.get)
                lang_str = "(" + language + ")"

            # Decode the audio
            options = whisper.DecodingOptions(task=task, language=language)
            result = whisper.decode(model, mel, options)
            print(f'{datetime.now().strftime("%H:%M:%S")} {lang_str} {result.text}')
            
        except Exception as e:
            print("Error", e)
                
    print("Stream ended")            
    process.wait()


if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('URL', type=str, help='Stream website and channel name, e.g. twitch.tv/forsen')
    parser.add_argument('--model', type=str, choices=['tiny','tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='small', help='Model to be used for generating audio transcription. Smaller models are faster and use less VRAM, but are also less accurate. .en models are more accuracte but only work on English audio.')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='translate', help='Whether to transcribe the audio (keep original language) or translate to english.')
    parser.add_argument('--language', type=str, default='auto', help='Language spoken in the stream. Default option is to auto detect the spoken language. See https://github.com/openai/whisper for available languages.')
    parser.add_argument('--interval', type=int, default='8', help='Interval between calls to the language model in seconds.')
    parser.add_argument('--directURL', action='store_true', help='Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to obtain the stream URL.')
    args = parser.parse_args()
    
    main(args)
