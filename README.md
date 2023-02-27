# stream-translator
Command line utility to transcribe or translate audio from livestreams in real time. Uses [streamlink](https://github.com/streamlink/streamlink) to 
get livestream URLs from various services and OpenAI's [whisper](https://github.com/openai/whisper) for transcription/translation.
This script is inspired by [audioWhisper](https://github.com/Awexander/audioWhisper) which transcribes/translates desktop audio.

## Prerequisites

1. [**Install and add ffmpeg to your PATH**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. [**Install CUDA on your system.**](https://developer.nvidia.com/cuda-downloads) If you installed a different version of CUDA than 11.3,
 change cu113 in requirements.txt accordingly. You can check the installed CUDA version with ```nvcc --version```.

## Setup

1. Setup a virtual environment.
2. ```git clone https://github.com/fortypercnt/stream-translator.git```
3. ```pip install -r requirements.txt```
4. Make sure that pytorch is installed with CUDA support. Whisper will probably not run in real time on a CPU.

## Command-line usage

```python translator.py URL --flags```

By default, the URL can be of the form ```twitch.tv/forsen``` and streamlink is used to obtain the .m3u8 link which is passed to ffmpeg.
See [streamlink plugins](https://streamlink.github.io/plugins.html) for info on all supported sites.


|             --flags             |     Default Value     |                                                                                                                       Description                                                                                                                        |
|:-------------------------------:|:---------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|            `--model`            |         small         |                                                                  Select model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                                                   |
|            `--task`             |       translate       |                                                                                    Whether to transcribe the audio (keep original language) or translate to english.                                                                                     |
|          `--language`           |         auto          |                                                           Language spoken in the stream. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                                           |
|          `--interval`           |           5           |                                                                                                 Interval between calls to the language model in seconds.                                                                                                 |
|     `--history_buffer_size`     |           0           | Seconds of previous audio/text to use for conditioning the model. Set to 0 to just use audio from the last interval. Note that this can easily lead to repetition/loops if the chosen language/model settings do not produce good results to begin with. |
|          `--beam_size`          |           5           |                                                                           Number of beams in beam search. Set to 0 to use greedy algorithm instead (faster but less accurate).                                                                           |
|           `--best_of`           |           5           |                                                                                              Number of candidates when sampling with non-zero temperature.                                                                                               |
|      `--preferred_quality`      |      audio_only       |                                                Preferred stream quality option. "best" and "worst" should always be available. Type "streamlink URL" in the console to see quality options for your URL.                                                 |
|         `--disable_vad`         |                       |                                                                                       Set this flag to disable additional voice activity detection by Silero VAD.                                                                                        |
|         `--direct_url`          |                       |                                                                        Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to obtain the stream URL.                                                                         |
|     `--use_faster_whisper`      |                       |                                                                             Set this flag to use faster_whisper implementation instead of the original OpenAI implementation                                                                             |
|  `--faster_whisper_model_path`  | whisper-large-v2-ct2/ |                                                                                        Path to a directory containing a Whisper model in the CTranslate2 format.                                                                                         |
|    `--faster_whisper_device`    |         cuda          |                                                                                                         Set the device to run faster-whisper on.                                                                                                         |
| `--faster_whisper_compute_type` |        float16        |                                                                Set the quantization type for faster_whisper. See [here](https://opennmt.net/CTranslate2/quantization.html) for more info.                                                                |

## Using faster-whisper

faster-whisper provides significant performance upgrades over the original OpenAI implementation (~ 4x faster, ~ 2x less memory).
To use it, follow the instructions [here](https://github.com/guillaumekln/faster-whisper#installation) to install faster-whisper and convert your models to CTranslate2 format.
Then you can run the CLI with --use_faster_whisper and set --faster_whisper_model_path to the location of your converted model.
