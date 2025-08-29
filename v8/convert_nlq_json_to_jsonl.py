#!/usr/bin/env python3
"""
Convert Ego4D NLQ JSON annotation files to flattened JSONL format.
Each line in the output JSONL will be a single language query, with video_uid, clip_uid, and all relevant fields.
"""
import json
import sys
from pathlib import Path

def flatten_nlq_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    with open(output_path, 'w') as out_f:
        for i, video in enumerate(data):
            if not isinstance(video, dict):
                print(f"[WARN] Skipping non-dict entry at index {i}: {type(video)}")
                continue
            video_uid = video.get('video_uid')
            for clip in video.get('clips', []):
                clip_uid = clip.get('clip_uid')
                for annotation in clip.get('annotations', []):
                    for query in annotation.get('language_queries', []):
                        flat = {
                            'video_uid': video_uid,
                            'clip_uid': clip_uid,
                        }
                        flat.update(query)
                        out_f.write(json.dumps(flat) + '\n')

if __name__ == "__main__":
    # Usage: python convert_nlq_json_to_jsonl.py <input_json> <output_jsonl>
    if len(sys.argv) != 3:
        print("Usage: python convert_nlq_json_to_jsonl.py <input_json> <output_jsonl>")
        sys.exit(1)
    flatten_nlq_json(sys.argv[1], sys.argv[2]) 