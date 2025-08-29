#!/usr/bin/env python3
"""
Split NLQ training data by video_uid: 80% of videos (and all their clips/queries) to train, 20% to validation.
Output the same structure as the input JSON, just with different videos in each split.
Also output a mapping (CSV and JSON) from video_uid to canonical_s3_location for each split, using manifest.csv.
"""

import json
import random
import os
import csv

def load_manifest(manifest_path):
    """Return a dict: video_uid -> canonical_s3_location"""
    mapping = {}
    with open(manifest_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Only keep rows that are videos (type==video or s3 path contains /video/)
            if row.get('type') == 'video' or '/video/' in row.get('canonical_s3_location', ''):
                mapping[row['file_uid']] = row['canonical_s3_location']
    return mapping

def split_nlq_by_video():
    nlq_train_path = "/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/annotations/nlq_train.json"
    manifest_path = "/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/annotations/manifest.csv"
    output_dir = "internvl3_data"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading NLQ training data...")
    with open(nlq_train_path, 'r') as f:
        data = json.load(f)

    videos = data['videos']
    print(f"Total videos: {len(videos)}")

    # Shuffle and split video_uids
    random.seed(42)
    video_indices = list(range(len(videos)))
    random.shuffle(video_indices)
    split_idx = int(0.8 * len(video_indices))
    train_indices = set(video_indices[:split_idx])
    val_indices = set(video_indices[split_idx:])

    train_videos = [videos[i] for i in train_indices]
    val_videos = [videos[i] for i in val_indices]

    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")

    # Write out new JSON files in the same structure as input
    def write_split(videos, out_path):
        split_data = {k: v for k, v in data.items() if k != 'videos'}
        split_data['videos'] = videos
        with open(out_path, 'w') as f:
            json.dump(split_data, f, indent=2)

    train_path = os.path.join(output_dir, "nlq_train_videosplit.json")
    val_path = os.path.join(output_dir, "nlq_val_videosplit.json")
    write_split(train_videos, train_path)
    write_split(val_videos, val_path)

    print(f"✅ Saved train split to: {train_path}")
    print(f"✅ Saved validation split to: {val_path}")

    # Update meta.json to use these new splits
    meta_path = os.path.join(output_dir, "meta.json")
    meta_data = {
        "train": "nlq_train_videosplit.json",
        "validation": "nlq_val_videosplit.json",
        "test": "nlq_val_videosplit.json"  # Use val as test for now
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f"✅ Updated meta.json to use video_uid splits")

    # --- Manifest mapping ---
    print("Loading manifest and creating video_uid to canonical_s3_location mapping...")
    manifest_map = load_manifest(manifest_path)

    def write_mapping(videos, split_name):
        mapping = {}
        for v in videos:
            video_uid = v['video_uid']
            s3 = manifest_map.get(video_uid, None)
            mapping[video_uid] = s3
        # Write JSON
        with open(os.path.join(output_dir, f"{split_name}_video_manifest.json"), 'w') as f:
            json.dump(mapping, f, indent=2)
        # Write CSV
        with open(os.path.join(output_dir, f"{split_name}_video_manifest.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_uid', 'canonical_s3_location'])
            for k, v in mapping.items():
                writer.writerow([k, v])
        print(f"✅ Saved {split_name} video manifest mapping (JSON and CSV)")

    write_mapping(train_videos, 'train')
    write_mapping(val_videos, 'val')

if __name__ == "__main__":
    split_nlq_by_video() 