# colab_whipser_stream_translator
使用colab云端运行直播实时语音转写  
具体使用方法请直接看文档 👇   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SEhfzUSm07IUjMd5_HrbmXd9cyh0N-wW?usp=sharing)  
有关发送到QQ频道使用了https://github.com/Mrs4s/go-cqhttp
## stream-translator
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


|         --flags         | Default Value |                                                                                                                       Description                                                                                                                        |
|:-----------------------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        `--model`        |     small     |                                                                  Select model size. See [here](https://github.com/openai/whisper#available-models-and-languages) for available models.                                                                   |
|        `--task`         |   translate   |                                                                                    Whether to transcribe the audio (keep original language) or translate to english.                                                                                     |
|      `--language`       |     auto      |                                                           Language spoken in the stream. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages.                                                           |
|      `--interval`       |       5       |                                                                                                 Interval between calls to the language model in seconds.                                                                                                 |
| `--history_buffer_size` |       0       | Seconds of previous audio/text to use for conditioning the model. Set to 0 to just use audio from the last interval. Note that this can easily lead to repetition/loops if the chosen language/model settings do not produce good results to begin with. |
|      `--beam_size`      |       5       |                                                                           Number of beams in beam search. Set to 0 to use greedy algorithm instead (faster but less accurate).                                                                           |
|       `--best_of`       |       5       |                                                                                              Number of candidates when sampling with non-zero temperature.                                                                                               |
|  `--preferred_quality`  |  audio_only   |                                                Preferred stream quality option. "best" and "worst" should always be available. Type "streamlink URL" in the console to see quality options for your URL.                                                 |
|     `--disable_vad`     |               |                                                                                       Set this flag to disable additional voice activity detection by Silero VAD.                                                                                        |
|     `--direct_url`      |               |                                                                        Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to obtain the stream URL.                                                                         |
