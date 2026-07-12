#!/usr/bin/env python3
"""
Merge diarized chunks with speaker name mapping.
"""

import json
import os
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_chunks.py <original_audio_file> <speaker_mapping.json>")
        print("\nExample speaker_mapping.json format:")
        print('{')
        print('  "chunk_01": {"SPEAKER_00": "William", "SPEAKER_01": "Neil", "SPEAKER_02": "Marcy"},')
        print('  "chunk_02": {"SPEAKER_00": "Marcy", "SPEAKER_01": "William", "SPEAKER_02": "Neil"},')
        print('  ...')
        print('}')
        sys.exit(1)

    audio_file = sys.argv[1]
    mapping_file = sys.argv[2]

    basename = Path(audio_file).stem
    output_dir = "./output"
    chunks_dir = os.path.join(output_dir, f"{basename}_chunks")

    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file '{mapping_file}' not found")
        sys.exit(1)

    if not os.path.exists(chunks_dir):
        print(f"Error: Chunks directory '{chunks_dir}' not found")
        print(f"Run diarize_chunked_separate.py first")
        sys.exit(1)

    # Load speaker mapping
    with open(mapping_file, 'r') as f:
        speaker_mapping = json.load(f)

    print(f"Loaded speaker mapping for {len(speaker_mapping)} chunks")

    # Merge all chunk text files
    merged_txt = os.path.join(output_dir, f"{basename}_merged.txt")
    merged_lines = []

    # Get all chunk files in order
    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.txt')])

    print(f"\nMerging {len(chunk_files)} chunks...")

    for chunk_file in chunk_files:
        chunk_name = chunk_file.replace('.txt', '')
        chunk_path = os.path.join(chunks_dir, chunk_file)

        if chunk_name not in speaker_mapping:
            print(f"Warning: No speaker mapping for {chunk_name}, skipping...")
            continue

        mapping = speaker_mapping[chunk_name]

        print(f"Processing {chunk_name}...")

        # Read chunk file and apply speaker mapping
        with open(chunk_path, 'r') as f:
            lines = f.readlines()

        # Skip header lines (first 4 lines)
        for line in lines[5:]:
            if line.strip():
                # Replace speaker labels
                updated_line = line
                for speaker_id, speaker_name in mapping.items():
                    updated_line = updated_line.replace(f"[{speaker_id}]", f"[{speaker_name}]")
                merged_lines.append(updated_line)

    # Save merged file
    with open(merged_txt, 'w') as f:
        f.writelines(merged_lines)

    print(f"\n✓ Merged transcript saved to: {merged_txt}")
    print(f"  Total lines: {len(merged_lines)}")

if __name__ == "__main__":
    main()
