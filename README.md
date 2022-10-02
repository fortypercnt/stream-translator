# stream-translator
Command line utility to transcribe or translate audio from livestreams in real time. Uses [streamlink](https://github.com/streamlink/streamlink) to 
get livestream URLs from various services and OpenAI's [Whisper](https://github.com/openai/whisper) for transcription/translation.
This script is inspired by [audioWhisper](https://github.com/Awexander/audioWhisper) which transcribes/translates desktop audio.

## Prerequisites

1. [**Install and add ffmpeg to your PATH**](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Click%20New%20and%20type%20the,Click%20OK%20to%20apply%20changes.)
2. [**Install CUDA**](https://developer.nvidia.com/cuda-downloads) If you installed a different version of CUDA than 11.3,
 change cu113 in requirements.txt accordingly. You can check the installed CUDA version with ```nvcc --version```.

## Setup

1. Setup a virtual environment.
2. ```git clone https://github.com/fortypercnt/stream-translator.git```.
3. ```pip install -r requirements.txt```.
4. Make sure that pytorch is installed with CUDA support. Whisper will probably not run in real time on a CPU.

## Command-line usage

```python translator.py URL --flags```
By default, the URL can be of the form ```twitch.tv/forsen``` and streamlink is used to obtain the .m3u8 link which is passed to ffmpeg.
See [streamlink plugins](https://streamlink.github.io/plugins.html) for info on all supported sites.

|      --flags          |  Default Value  |      Description                                                                                                                     |
|:---------------------:|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------:|
|`--model`              | small           | Select model size. See [here](https://github.com/openai/whisper#available-models-and-languages)                                      |
|`--task`               | transcribe      | Whether to transcribe the audio (keep original language) or translate to english.                                                    |
|`--language`           | auto            | Language spoken in the stream. See [here](https://github.com/openai/whisper#available-models-and-languages) for available languages. |
|`--interval`           | 8               | Interval between calls to the language model in seconds.                                                                             |
|`--directURL`          |                 | Set this flag to pass the URL directly to ffmpeg. Otherwise, streamlink is used to obtain the stream URL.                            |