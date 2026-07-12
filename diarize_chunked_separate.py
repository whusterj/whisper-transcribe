#!/usr/bin/env python3
"""
Run speaker diarization on long audio files by processing in chunks.
Outputs each chunk separately so speaker labels can be manually identified.
"""

import json
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.audio import load_audio

# Chunk size in seconds (30 minutes)
CHUNK_SIZE_SECONDS = 1800

def chunk_audio(audio, sample_rate=16000, chunk_size_seconds=CHUNK_SIZE_SECONDS):
    """Split audio into chunks."""
    chunk_size_samples = chunk_size_seconds * sample_rate
    num_chunks = int(np.ceil(len(audio) / chunk_size_samples))

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size_samples
        end = min((i + 1) * chunk_size_samples, len(audio))
        chunks.append({
            'audio': audio[start:end],
            'start_time': start / sample_rate,
            'end_time': end / sample_rate,
            'chunk_idx': i
        })

    return chunks

def filter_segments_by_time(segments, start_time, end_time):
    """Filter segments that overlap with the given time range."""
    filtered = []
    for seg in segments:
        # Check if segment overlaps with chunk time range
        if seg['end'] >= start_time and seg['start'] <= end_time:
            filtered.append(seg)
    return filtered

def main():
    if len(sys.argv) < 2:
        print("Usage: python diarize_chunked_separate.py <audio_file>")
        print("\nOutputs separate files for each chunk so you can identify speakers.")
        sys.exit(1)

    audio_file = sys.argv[1]
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)

    basename = Path(audio_file).stem
    output_dir = "./output"
    json_file = os.path.join(output_dir, f"{basename}.json")

    if not os.path.exists(json_file):
        print(f"Error: Transcription not found at {json_file}")
        print("Run ./transcribe-no-diarize.sh first")
        sys.exit(1)

    print(f"Found existing transcription: {json_file}")
    print("Loading transcription...")

    with open(json_file, 'r') as f:
        result = json.load(f)

    print(f"Loaded {len(result.get('segments', []))} segments")

    # Load audio
    print("\nLoading audio...")
    audio = load_audio(audio_file)
    duration_seconds = len(audio) / 16000
    duration_hours = int(duration_seconds // 3600)
    duration_minutes = int((duration_seconds % 3600) // 60)
    print(f"Audio duration: {duration_hours}h {duration_minutes}m")

    # Split into chunks
    print(f"\nSplitting audio into {CHUNK_SIZE_SECONDS/60:.0f}-minute chunks...")
    chunks = chunk_audio(audio, chunk_size_seconds=CHUNK_SIZE_SECONDS)
    print(f"Created {len(chunks)} chunks")

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load diarization model
    print("Loading diarization model...")
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=torch.device(device)
    )

    min_speakers = 2
    max_speakers = 4

    # Create chunks output directory
    chunks_dir = os.path.join(output_dir, f"{basename}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Process each chunk and save separately
    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i+1:02d}"
        print(f"\n{'='*60}")
        print(f"Processing {chunk_name} ({chunk['start_time']/60:.0f}-{chunk['end_time']/60:.0f} min)...")
        print(f"{'='*60}")

        # Run diarization on this chunk
        chunk_diarize_df = diarize_model(
            chunk['audio'],
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # Adjust timestamps to absolute time
        chunk_diarize_df['start'] = chunk_diarize_df['start'] + chunk['start_time']
        chunk_diarize_df['end'] = chunk_diarize_df['end'] + chunk['start_time']

        print(f"  Found {len(chunk_diarize_df)} speaker segments")

        # Get unique speakers in this chunk
        speakers = chunk_diarize_df['speaker'].unique()
        print(f"  Speakers in this chunk: {', '.join(speakers)}")

        # Filter transcription segments for this chunk
        chunk_segments = filter_segments_by_time(
            result['segments'],
            chunk['start_time'],
            chunk['end_time']
        )

        chunk_result = {'segments': chunk_segments}

        # Assign speakers to this chunk's segments
        chunk_result = assign_word_speakers(chunk_diarize_df, chunk_result)

        # Save chunk transcript as text file
        txt_file = os.path.join(chunks_dir, f"{chunk_name}.txt")
        with open(txt_file, 'w') as f:
            f.write(f"=== Chunk {i+1}/{len(chunks)} ===\n")
            f.write(f"Time: {chunk['start_time']/60:.0f}-{chunk['end_time']/60:.0f} minutes\n")
            f.write(f"Speakers found: {', '.join(speakers)}\n")
            f.write(f"{'='*60}\n\n")

            for segment in chunk_result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"]
                timestamp = f"{segment['start']/60:.1f}min"
                f.write(f"[{timestamp}] [{speaker}]: {text}\n")

        print(f"  ✓ Saved: {txt_file}")

        # Save chunk metadata
        meta_file = os.path.join(chunks_dir, f"{chunk_name}_metadata.json")
        with open(meta_file, 'w') as f:
            json.dump({
                'chunk_index': i + 1,
                'total_chunks': len(chunks),
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'speakers': list(speakers),
                'num_segments': len(chunk_result['segments'])
            }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All chunks processed!")
    print(f"{'='*60}")
    print(f"\nChunk files saved to: {chunks_dir}/")
    print(f"\nNext steps:")
    print(f"1. Review each chunk_XX.txt file")
    print(f"2. Identify which SPEAKER_XX corresponds to which person in each chunk")
    print(f"3. Create a mapping file (speaker_mapping.json) with this format:")
    print(f"   {{")
    print(f"     \"chunk_01\": {{\"SPEAKER_00\": \"William\", \"SPEAKER_01\": \"Neil\", ...}},")
    print(f"     \"chunk_02\": {{\"SPEAKER_00\": \"Marcy\", \"SPEAKER_01\": \"William\", ...}},")
    print(f"     ...")
    print(f"   }}")
    print(f"4. Run the merge script (TODO: create this) to combine chunks with correct names")

if __name__ == "__main__":
    main()
