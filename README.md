# Offline Audio Transcription with WhisperX

This tool provides offline transcription of audio recordings using [WhisperX](https://github.com/m-bain/whisperX), an enhanced version of OpenAI's Whisper that adds:
- Word-level timestamps with accurate alignment
- Speaker diarization (identifying who spoke when)
- Better performance with batched inference

Perfect for transcribing meetings, interviews, and other recordings without uploading to third-party services.

## Installation

This installation has been tested with `Python 3.10` on Ubuntu 20.04 in WSL2. It should work on other platforms as well.

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

**Note:** This installs PyTorch 2.8 with CUDA 12.8 support. The installation includes patches for compatibility with pyannote.audio models used for speaker diarization.

### HuggingFace Token (Required for Diarization)

To use speaker diarization features:

1. Create a HuggingFace account at https://huggingface.co

2. **Accept user agreements for the two required gated models** (click "Agree and access repository" on each page):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

3. Get your access token from https://huggingface.co/settings/tokens

4. Set it as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

Alternatively, copy `.env.example` to `.env` and fill in your token, then source it before running:
```bash
cp .env.example .env
# Edit .env with your token
source .env
```

**Important:** You must accept BOTH model agreements before diarization will work!

## Usage

For long recordings (>1 hour), use the **three-step workflow** to avoid memory issues and ensure consistent speaker labels:

### Step 1: Transcribe Without Diarization

```bash
./transcribe-no-diarize.sh "meeting-recording.m4a"
```

This transcribes and aligns the audio without speaker diarization (lower memory usage). Output saved to `./output/`.

### Step 2: Diarize in Chunks

```bash
source .env  # Make sure HF_TOKEN is set
python diarize_chunked_separate.py "meeting-recording.m4a"
```

This processes diarization in 30-minute chunks and saves each chunk separately to `./output/meeting-recording_chunks/`. Each chunk file shows which `SPEAKER_XX` labels appear in that segment.

### Step 3: Create Speaker Mapping and Merge

1. Review each `chunk_XX.txt` file in the chunks directory
2. Identify which `SPEAKER_XX` corresponds to which person in each chunk (labels may differ between chunks)
3. Create `speaker_mapping.json`:

```json
{
  "chunk_01": {"SPEAKER_00": "William", "SPEAKER_01": "Neil", "SPEAKER_02": "Marcy"},
  "chunk_02": {"SPEAKER_00": "Marcy", "SPEAKER_01": "William", "SPEAKER_02": "Neil"},
  "chunk_03": {"SPEAKER_00": "William", "SPEAKER_01": "Marcy", "SPEAKER_02": "Neil"}
}
```

4. Merge with correct names:

```bash
python merge_chunks.py "meeting-recording.m4a" speaker_mapping.json
```

This creates `./output/meeting-recording_merged.txt` with all speaker names correctly applied!

### Default Settings

The transcription script uses these defaults:
- **Model:** `large-v3` (most accurate)
- **Language:** English
- **Output formats:** All formats (JSON, SRT, VTT, TXT, TSV)
- **Output directory:** `./output/`

Diarization settings:
- **Chunk size:** 30 minutes (configurable in script)
- **Min speakers:** 2
- **Max speakers:** 4

### Output Files

Transcription results will be saved in the `./output/` directory with multiple formats:
- `.json` - Full transcription with word-level timestamps and speaker labels
- `.srt` - Subtitle format for video
- `.vtt` - WebVTT format for web videos
- `.txt` - Plain text transcription
- `.tsv` - Tab-separated values with timestamps

### Customization

To modify the default settings, edit `transcribe-no-diarize.sh`. Common options:
- `--model` - Choose model size: `tiny`, `base`, `small`, `medium`, `large-v3`
- `--min_speakers` / `--max_speakers` - Adjust expected number of speakers
- `--language` - Set language code (e.g., `es`, `fr`, `de`)
- Remove `--diarize` if you don't need speaker identification

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

### Why Three Steps?

**Memory limitations:** Speaker diarization models are extremely memory-intensive. Even high-end GPUs (like RTX 4090) can run out of VRAM on long recordings when running transcription + diarization together.

**Speaker consistency:** Chunked diarization assigns speaker labels independently per chunk. `SPEAKER_00` in chunk 1 might be a different person than `SPEAKER_00` in chunk 2. The separate-chunks approach lets you identify speakers correctly for each chunk, then merge them with consistent names.

**Tips:**
- For shorter recordings (<30 min), you might be able to run all steps together
- Adjust chunk size in `diarize_chunked_separate.py` if needed (larger = fewer chunks but more memory)
- Use a smaller Whisper model (`medium` or `base`) if transcription runs out of memory
