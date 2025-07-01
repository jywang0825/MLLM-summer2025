import json
import os
import csv

def load_large_json_or_jsonl(path):
    """Load the narrations file with the known Ego4D structure."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# Paths
NLQ_PATH = os.path.join(os.path.dirname(__file__), 'data_files', 'nlq_train.json')
NARRATION_PATH = os.path.join(os.path.dirname(__file__), 'data_files', 'all_narrations_redacted.json')
MANIFEST_PATH = '/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/clips/manifest.csv'
OUTPUT_PATH = 'nlq_clip_narrations.json'

# Load NLQ annotation file
with open(NLQ_PATH, 'r') as f:
    nlq_data = json.load(f)

# Load narrations (Ego4D structure)
narr_data = load_large_json_or_jsonl(NARRATION_PATH)

# Build a mapping from canonical video_uid to list of narrations (sorted by time)
video_to_narrs = {}
for video_uid, video_info in narr_data['videos'].items():
    narrs = video_info.get('narrations', [])
    narr_entries = []
    for narr in narrs:
        narr_time = narr.get('time')
        narr_text = narr.get('text')
        if narr_time is None or narr_text is None:
            continue
        narr_entries.append({
            'narration_time_sec': narr_time,
            'narration_text': narr_text
        })
    narr_entries.sort(key=lambda x: x['narration_time_sec'])
    video_to_narrs[video_uid] = narr_entries

# Load manifest.csv and build a mapping from exported_clip_uid to (parent_video_uid, parent_start_sec)
man_clip_map = {}
with open(MANIFEST_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        exported_clip_uid = row['exported_clip_uid']
        parent_video_uid = row['parent_video_uid']
        try:
            parent_start_sec = float(row['parent_start_sec'])
            parent_end_sec = float(row['parent_end_sec'])
        except Exception:
            parent_start_sec = None
            parent_end_sec = None
        man_clip_map[exported_clip_uid] = {
            'parent_video_uid': parent_video_uid,
            'parent_start_sec': parent_start_sec,
            'parent_end_sec': parent_end_sec
        }

# For each NLQ clip, extract narrations in the correct canonical video/time range
results = []
for i, video in enumerate(nlq_data['videos']):
    if i % 10 == 0:
        print(f"Processing video {i+1}/{len(nlq_data['videos'])}")
    for clip in video['clips']:
        clip_uid = clip['clip_uid']
        # Find mapping in manifest
        manifest_entry = man_clip_map.get(clip_uid)
        if not manifest_entry:
            continue
        parent_video_uid = manifest_entry['parent_video_uid']
        parent_start_sec = manifest_entry['parent_start_sec']
        if parent_video_uid is None or parent_start_sec is None:
            continue
        # Calculate canonical video time range
        clip_start = clip['clip_start_sec'] + parent_start_sec
        clip_end = clip['clip_end_sec'] + parent_start_sec
        narrs = video_to_narrs.get(parent_video_uid, [])
        # Find narrations within [clip_start, clip_end]
        clip_narrs = [n for n in narrs if clip_start <= n['narration_time_sec'] <= clip_end]
        results.append({
            'clip_uid': clip_uid,
            'parent_video_uid': parent_video_uid,
            'clip_start_sec': clip_start,
            'clip_end_sec': clip_end,
            'captions': clip_narrs
        })

# Save to output JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} clips with narrations to {OUTPUT_PATH}") 