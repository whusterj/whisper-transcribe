# Offline Audio Transcription with Whisper AI

Here's an example of how to use OpenAI's Whisper package to make OFFLINE transcriptions of your recordings of speech. This is useful for personal recordings that you want to transcribe, but would rather not upload to a third-party transcription service.

## Installation

This installation has been tested with `Python 3.10.12` on Ubuntu 20.02. It should work on other Platforms as well, and OpenAI says Whisper should work with all Python versions 3.9 to 3.11.

First, you will need `ffmpeg` on your system, if you don't have it already:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

For other platforms, see the [Whisper GitHub repo][1].

Now you can install the python requirements. Create a virtual environment, activate it, and pip install from requirements.txt:

```bash
python -m venv .venv/
source .venv/bin/activate
pip install -r requirements.txt
```

If you encounter errors during installation, you may need to install `rust` on your platform as well.

## Run It!

To transcribe an MP3 to a TXT file right away, do this:

```bash
./transcribe.sh my.mp3
```

The scripts `wav2mp3.sh` and `transcribe.sh` are provided here for convenience.

`wav2mp3.sh` will take a WAV file and convert it to MP3 using ffmpeg. If you already have an MP3 file, then this is not needed. Whisper seems to perform much better when given an MP3 file, probably because of the smaller file size.

`transcribe.sh` is where the magic happens. This is a basic Whisper command that accepts an MP3 file as an argument. The transcription will be printed to stdio and a text file. For efficiency, this script uses the "Small English" (small.en) training model. This model is a little more than 400MB download. It should download itself automatically on the first run of the script. It seems more than adequate for transcribing clean audio in the English language. Whisper support many other languages as well and have large models that promise more accurate transcription.

This covers the most basic use case of transcribing audio to a text file, but Whisper is very powerful! For example, it can be used to generate time-coded .SRT video caption files and it can also translate to other languages as it transcribes. It also has a Python API enabling more robust use cases and automation. See the [GitHub Show and Tell Page][3] or `whisper -h` help command for more details and options.

## The Key Package: OpenAI Whisper

The [Whisper GitHub repo][1] contains the installation and usage instructions. As of July 2023, it can be installed with `pip`.

```bash
pip install -U openai-whisper
```

NOTE: the [whisper][2] package is something else entirely: "Whisper is a fixed-size database, similar in design and purpose to RRD (round-robin-database)." So make sure you are installing OpenAI's transcription package!

[1]: https://github.com/openai/whisper "OpenAI Whisper GitHub Repo"
[2]: https://pypi.org/project/whisper/ "PyPi Whisper Package"
[3]: https://github.com/openai/whisper/discussions/categories/show-and-tell "Whisper Show and Tell"
