# Offline Audio Transcription with WhisperX

This tool provides offline transcription of audio recordings using [WhisperX](https://github.com/m-bain/whisperX), an enhanced version of OpenAI's Whisper that adds:
- Word-level timestamps with accurate alignment
- Speaker diarization (identifying who spoke when)
- Better performance with batched inference

Perfect for transcribing meetings, interviews, and other recordings without uploading to third-party services.

## Installation

Tested with `Python 3.10` on Ubuntu 20.04 in WSL2, with an NVIDIA GPU.

`requirements.txt` was pinned on that machine, so it names the CUDA 12.8 wheels.
Those carry a `platform_system == "Linux"` marker, so pip skips them elsewhere
and `torch` resolves to the wheel for the host. The same file therefore installs
on Linux and on macOS.

**On macOS the work runs on the CPU.** `diarize.py` asks for CUDA and falls back
to the CPU when it is absent, and neither WhisperX nor pyannote.audio is
dependable on Apple's MPS backend. Transcribing a long recording that way takes
hours rather than minutes, so treat macOS as somewhere to read and try the code,
not somewhere to process a meeting.

### Prerequisites

First, you will need `ffmpeg` on your system:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

### Python Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** This installs PyTorch 2.8 with CUDA 12.8 support. No patch is necessary: `pyannote.audio` 4 reads its checkpoints in a format that the `weights_only` default of PyTorch 2.6 and later accepts.

`whisperx` is pinned to the release candidate `3.8.7rc1`. Its code is identical to the `3.8.6` release, but `3.8.6` pins `huggingface-hub<1.0.0`, which holds `transformers` on the 4.x line and keeps three security alerts open. See the header of `requirements.txt`.

### HuggingFace Token (Required for Diarization)

To use speaker diarization features:

1. Create a HuggingFace account at https://huggingface.co

2. **Accept the user agreement for the gated diarization model** (click "Agree and access repository"):
   - https://huggingface.co/pyannote/speaker-diarization-community-1

3. Get your access token from https://huggingface.co/settings/tokens

4. Copy `.env.example` to `.env` and fill in your token:

```bash
cp .env.example .env
# Edit .env with your token
source .env
```

Or export it directly:

```bash
export HF_TOKEN="your_token_here"
```

**Important:** You must accept the model agreement before diarization will work.

### Telemetry

`pyannote.audio` 4 reports anonymous usage to `pyannote.ai` by default: the origin of the pipeline, its Python class, the duration of each file, and the requested speaker counts. `diarize.py` turns this off. To send the metrics, export `PYANNOTE_METRICS_ENABLED=1` before you run it.

## Usage

For long recordings (>1 hour), use the **two-step workflow** to avoid memory issues. Running transcription and diarization together on a single GPU call can exhaust VRAM on recordings of this length.

### Step 1: Transcribe Without Diarization

```bash
./transcribe-no-diarize.sh "meeting-recording.m4a"
```

This transcribes and aligns the audio without speaker diarization (lower memory usage). Output saved to `./.output/`.

### Step 2: Diarize

```bash
source .env  # Make sure HF_TOKEN is set
python diarize.py "meeting-recording.m4a" 3 3
```

The second and third arguments are the minimum and maximum number of expected speakers. Setting them equal (e.g. `3 3`) when you know the exact count gives the best results. Diarization runs on the **full audio in a single pass**, so speaker labels (`SPEAKER_00`, `SPEAKER_01`, etc.) are consistent throughout the entire transcript.

Output is saved to `.output/meeting-recording_diarized.txt`.

#### Automatic speaker identification with reference audio (optional)

If you have a clean recording of one or more speakers (30+ seconds of solo speech), you can pass them as reference audio to automatically name the matching cluster:

```bash
python diarize.py "meeting-recording.m4a" 3 3 \
  --ref Alice:".data/alice-sample.m4a" \
  --ref Bob:".data/bob-sample.m4a"
```

The script extracts speaker embeddings from each reference clip, compares them against the clusters found in the meeting, and renames the best-matching labels. Any unmatched speakers retain their `SPEAKER_XX` label for manual renaming.

For speakers without reference audio, rename manually with `sed`:

```bash
sed -i 's/\[SPEAKER_02\]/[Charlie]/g' .output/meeting-recording_diarized.txt
```

**Note:** Same-gender speakers with similar voices will have some cross-contamination even with reference audio. Reference clips reduce errors but don't eliminate them entirely.

### Default Settings

The transcription script uses these defaults:
- **Model:** `large-v3` (most accurate)
- **Language:** English
- **Output formats:** All formats (JSON, SRT, VTT, TXT, TSV)
- **Output directory:** `./.output/`

Diarization defaults (configurable as CLI args):
- **Min speakers:** 2
- **Max speakers:** 4

### Output Files

Transcription results will be saved in the `./.output/` directory:
- `.json` — Full transcription with word-level timestamps (used by diarization step)
- `.srt` — Subtitle format for video
- `.vtt` — WebVTT format for web videos
- `.txt` — Plain text transcription
- `.tsv` — Tab-separated values with timestamps
- `_diarized.txt` — Final transcript with speaker labels

### Customization

To modify the default settings, edit `transcribe-no-diarize.sh`. Common options:
- `--model` — Choose model size: `tiny`, `base`, `small`, `medium`, `large-v3`
- `--language` — Set language code (e.g., `es`, `fr`, `de`)

See `whisperx --help` for all available options.

## About WhisperX

WhisperX builds on OpenAI's Whisper with several improvements:
- **Faster:** Batched inference using faster-whisper backend
- **More accurate:** Uses forced phoneme alignment for precise word timestamps
- **Speaker labels:** Integrates pyannote.audio for speaker diarization
- **Better formatting:** Improved sentence segmentation

For more information, see the [WhisperX GitHub repository](https://github.com/m-bain/whisperX).

## Troubleshooting

### PyTorch Compatibility Warnings

You may see warnings about version mismatches between the training environment and your current setup. These are generally safe to ignore as long as the transcription completes successfully.

### GPU Not Detected

If you see ONNX Runtime GPU discovery warnings in WSL2, this is normal. WhisperX will still use your GPU via CUDA.

### Speaker Count Hints Ignored

Older version combinations silently ignored `min_speakers`/`max_speakers`. The pinned versions obey them. Do not go back to `whisperx==3.3.1` and `pyannote.audio==3.3.2` to correct speaker splits: those releases carry the `transformers` and `huggingface-hub` versions that this upgrade removed.

### Open torch Alerts

Three Dependabot alerts on `torch` stay open. `whisperx` pins `torch~=2.8.0`, so nothing above 2.8.x can install. All three are memory-corruption bugs in `torch.jit.script`, `torch.lstm_cell`, and `torch.nn.utils.rnn.unpack_sequence`, which this code does not call. Bump `torch` and `torchaudio` when an upstream `whisperx` release widens that pin.

### Open lightning Alert

**Load diarization checkpoints only from the pinned `pyannote/*` repositories on Hugging Face.** Dependabot alert 74 (GHSA-qqmf-gpg7-g8gw) covers `lightning` 2.6.5, the latest release, and a checkpoint file from an untrusted source can run arbitrary code through `load_from_checkpoint`, which `pyannote.audio` calls.
