#!/bin/bash

# Transcription WITHOUT diarization (saves memory)
# Usage: ./transcribe-no-diarize.sh <audio_file>
# Run `python diarize.py <audio_file>` afterwards to add speaker labels

if [ -z "$1" ]; then
    echo "Usage: $0 <audio_file>"
    echo "Example: $0 \"meeting-2026-01-10.m4a\""
    echo ""
    echo "This transcribes without diarization (lower memory usage)."
    echo "Afterwards, add speaker labels with: python diarize.py <audio_file>"
    exit 1
fi

whisperx "$1" \
    --model large-v3 \
    --language en \
    --output_dir ./.output \
    --output_format all \
    --batch_size 16
