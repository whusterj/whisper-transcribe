#!/usr/bin/env python3
"""
Run speaker diarization on a pre-transcribed audio file.
Requires a JSON transcription from transcribe-no-diarize.sh.

Usage:
  python diarize.py <audio_file> [min_speakers [max_speakers]] [--ref NAME:file ...]

  --ref NAME:file    Reference audio for a known speaker. Can be repeated.
                     Example: --ref Alice:".data/alice-sample.m4a"
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from pyannote.audio import Inference
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.audio import load_audio


def cosine_similarity(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-8 else 0.0


def get_inference(diarize_model, hf_token, device):
    """Get a speaker embedding inference model, reusing the pipeline's if possible."""
    try:
        embedding_model = diarize_model.model.embedding
        inference = Inference(embedding_model, window="whole")
    except AttributeError:
        from pyannote.audio import Model
        embedding_model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            use_auth_token=hf_token,
        )
        inference = Inference(embedding_model, window="whole")
    inference.to(torch.device(device))
    return inference


def extract_embedding(inference, audio, sample_rate=16000):
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    return np.array(inference({"waveform": waveform, "sample_rate": sample_rate})).flatten()


def get_cluster_embeddings(inference, audio, diarize_df, sample_rate=16000):
    """Extract a mean embedding per detected speaker cluster."""
    embeddings = {}
    for speaker in diarize_df["speaker"].unique():
        segs = diarize_df[diarize_df["speaker"] == speaker]
        chunks = []
        for _, row in segs.iterrows():
            start = int(row["start"] * sample_rate)
            end = int(row["end"] * sample_rate)
            if end - start > sample_rate:  # skip segments < 1s
                chunks.append(audio[start:end])
        if not chunks:
            continue
        # Use up to 60s (longest chunks first) for a stable embedding
        chunks.sort(key=len, reverse=True)
        combined = np.concatenate(chunks[:30])[: sample_rate * 60]
        embeddings[speaker] = extract_embedding(inference, combined, sample_rate)
    return embeddings


def match_to_references(cluster_embeddings, reference_audios, inference, sample_rate=16000):
    """Greedily match speaker clusters to named references by cosine similarity."""
    ref_embeddings = {}
    for name, path in reference_audios.items():
        print(f"  Loading reference audio for {name}: {path}")
        ref_audio = load_audio(path)
        ref_embeddings[name] = extract_embedding(inference, ref_audio, sample_rate)
        print(f"    {len(ref_audio) / sample_rate:.1f}s of reference audio")

    candidates = [
        (cosine_similarity(c_emb, r_emb), speaker, name)
        for speaker, c_emb in cluster_embeddings.items()
        for name, r_emb in ref_embeddings.items()
    ]
    candidates.sort(reverse=True)

    label_map = {}
    assigned_speakers, assigned_names = set(), set()
    for sim, speaker, name in candidates:
        if speaker not in assigned_speakers and name not in assigned_names:
            print(f"  Matched {speaker} -> {name} (cosine similarity: {sim:.3f})")
            label_map[speaker] = name
            assigned_speakers.add(speaker)
            assigned_names.add(name)

    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument("min_speakers", type=int, nargs="?", default=2)
    parser.add_argument("max_speakers", type=int, nargs="?", default=4)
    parser.add_argument(
        "--ref",
        action="append",
        metavar="NAME:file",
        default=[],
        help="Reference audio for a known speaker",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)

    basename = Path(args.audio_file).stem
    output_dir = "./.output"
    json_file = os.path.join(output_dir, f"{basename}.json")

    if not os.path.exists(json_file):
        print(f"Error: Transcription not found at {json_file}")
        print("Run ./transcribe-no-diarize.sh first")
        sys.exit(1)

    reference_audios = {}
    for ref in args.ref:
        if ":" not in ref:
            print(f"Error: --ref must be NAME:file, got: {ref}")
            sys.exit(1)
        name, path = ref.split(":", 1)
        if not os.path.exists(path):
            print(f"Error: Reference file not found: {path}")
            sys.exit(1)
        reference_audios[name] = path

    print(f"Loading transcription: {json_file}")
    with open(json_file, "r") as f:
        result = json.load(f)
    print(f"Loaded {len(result.get('segments', []))} segments")

    print("Loading audio...")
    audio = load_audio(args.audio_file)
    duration_seconds = len(audio) / 16000
    print(f"Audio duration: {int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading diarization model...")
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=torch.device(device),
    )

    print(f"Running diarization (min_speakers={args.min_speakers}, max_speakers={args.max_speakers})...")
    print("Diarization runs on the full audio in one pass — this may take a while.")
    diarize_segments = diarize_model(
        audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    speakers = diarize_segments["speaker"].unique()
    print(f"Found {len(speakers)} speakers: {', '.join(sorted(speakers))}")

    label_map = {s: s for s in speakers}
    if reference_audios:
        print("\nMatching speakers to reference audio...")
        inference = get_inference(diarize_model, hf_token, device)
        cluster_embeddings = get_cluster_embeddings(inference, audio, diarize_segments)
        matches = match_to_references(cluster_embeddings, reference_audios, inference)
        label_map.update(matches)

    print("\nAssigning speakers to transcript segments...")
    result = assign_word_speakers(diarize_segments, result)

    txt_file = os.path.join(output_dir, f"{basename}_diarized.txt")
    with open(txt_file, "w") as f:
        for segment in result["segments"]:
            raw_speaker = segment.get("speaker", "UNKNOWN")
            speaker = label_map.get(raw_speaker, raw_speaker)
            text = segment["text"].strip()
            timestamp = f"{segment['start'] / 60:.1f}min"
            f.write(f"[{timestamp}] [{speaker}]: {text}\n")

    print(f"\nDiarized transcript saved to: {txt_file}")
    if reference_audios:
        unmatched = [s for s in speakers if label_map.get(s) == s]
        if unmatched:
            print(f"Unmatched speakers (rename manually): {', '.join(sorted(unmatched))}")
    else:
        print("Tip: use --ref to auto-identify speakers from reference audio:")
        print(f'  python diarize.py "{args.audio_file}" {args.min_speakers} {args.max_speakers} --ref Alice:.data/alice-sample.m4a')


if __name__ == "__main__":
    main()
