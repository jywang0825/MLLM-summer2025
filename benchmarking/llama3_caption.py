import json
import sqlite3
import requests
import os
from datetime import datetime
import cv2
import base64
import time
import random
import math

# Database setup
def setup_database():
    pass  # No longer needed

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_random_frames(video_path, output_dir, start_sec, end_sec, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    end_sec = min(end_sec, duration)
    if end_sec <= start_sec:
        print(f"[WARNING] Invalid interval: start_sec={start_sec}, end_sec={end_sec}")
        cap.release()
        return []
    # Randomly select unique timestamps
    possible_times = list(range(int(start_sec), int(end_sec)))
    if len(possible_times) < num_frames:
        selected_times = possible_times
    else:
        selected_times = sorted(random.sample(possible_times, num_frames))
    frame_paths = []
    for idx, sec in enumerate(selected_times):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame at {sec}s in {video_path}")
            continue
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{idx:02d}_{sec}s.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append((frame_path, sec))
    cap.release()
    print(f"[INFO] Extracted {len(frame_paths)} random frames from {video_path}")
    return frame_paths

def caption_frame(img_b64, video_name):
    url = "http://localhost:11434/api/generate"
    prompt = f"Very shortly in a sentence describe the main action in this video frame from {video_name}. Also ignore any background information"
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start
        print("[DEBUG] Ollama response:", response.status_code, response.text)
        if response.status_code == 200:
            return response.json().get("response"), elapsed
        else:
            print(f"[ERROR] Ollama API error: {response.status_code} {response.text}")
            return None, elapsed
    except Exception as e:
        print(f"[ERROR] Exception during Ollama request: {e}")
        return None, 0

def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            return frame_count / fps
        else:
            return None
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None

# Load NLQ annotations
def load_nlq_data(file_path):
    with open(file_path) as f:
        nlq_data = json.load(f)
    
    # Extract video and clip information
    videos_info = []
    for video in nlq_data['videos']:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            videos_info.append({
                'video_uid': video_uid,
                'clip_uid': clip_uid,
                'clip_start_sec': clip.get('clip_start_sec'),
                'clip_end_sec': clip.get('clip_end_sec')
            })
    
    return videos_info

# Find video file path based on video_uid
def find_video_path(video_uid, clips_directory):
    """Find the video file path based on video_uid"""
    # Look for video file with the video_uid as filename
    possible_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for ext in possible_extensions:
        video_path = os.path.join(clips_directory, f"{video_uid}{ext}")
        if os.path.exists(video_path):
            return video_path
    
    # If not found, return None
    return None

def extract_frames_at_timestamps(video_path, output_dir, timestamps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths = []
    for idx, sec in enumerate(timestamps):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame at {sec}s in {video_path}")
            continue
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{idx:02d}_{int(sec)}s.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append((frame_path, sec))
    cap.release()
    print(f"[INFO] Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths

# Main caption generation and JSON output
def main():
    clips_directory = '/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/clips'
    print("Using clips_directory:", clips_directory)
    nlq_json_path = os.path.join(os.path.dirname(__file__), '..', 'data_files', 'nlq_train.json')
    output_json = 'llama3_nlq_frame_captions.json'  # Unique output file for llama benchmarking
    possible_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    frames_dir = 'llama3_extracted_frames'  # Unique frames directory for llama benchmarking
    os.makedirs(frames_dir, exist_ok=True)

    with open(nlq_json_path) as f:
        nlq_data = json.load(f)

    # Load existing results if output_json exists
    results = []
    processed_clip_uids = set()
    if os.path.exists(output_json):
        with open(output_json) as f:
            try:
                results = json.load(f)
                processed_clip_uids = set(r['clip_uid'] for r in results)
                print(f"[INFO] Loaded {len(results)} already captioned clips from {output_json}")
            except Exception as e:
                print(f"[WARNING] Could not load existing results: {e}")

    total_clips = sum(len(v['clips']) for v in nlq_data['videos'])
    print(f"Found {total_clips} NLQ clips to process")
    processed = len(processed_clip_uids)
    clip_counter = 1
    max_videos = 5
    video_count = 0

    NARRATIONS_JSON_PATH = '/shared/ssd_14T/home/wangj/your-repo/finetuning/data_files/nlq_clip_narrations.json'
    with open(NARRATIONS_JSON_PATH) as f:
        narrations_data = json.load(f)
    # Build a mapping from clip_uid to list of narration_time_sec
    clip_to_caption_times = {
        entry['clip_uid']: [c['narration_time_sec'] for c in entry.get('captions', [])]
        for entry in narrations_data
    }

    for video in nlq_data['videos']:
        if video_count >= max_videos:
            print(f"[INFO] Reached the limit of {max_videos} videos. Stopping.")
            break
        video_uid = video['video_uid']
        video_results = []
        for clip in video['clips']:
            print(f"[INFO] Processing clip {clip_counter}/{total_clips}")
            clip_counter += 1
            clip_uid = clip['clip_uid']
            if clip_uid in processed_clip_uids:
                print(f"[INFO] Skipping already captioned clip {clip_uid}")
                continue
            start_sec = clip.get('clip_start_sec', 0)
            end_sec = clip.get('clip_end_sec', 0)
            video_path = None
            for ext in possible_extensions:
                candidate = os.path.join(clips_directory, f"{clip_uid}{ext}")
                if os.path.exists(candidate):
                    print(f"[INFO] Found video file: {candidate}")
                    video_path = candidate
                    break
            if not video_path:
                print(f"[WARNING] No video file found for clip {clip_uid}")
                continue
            print(f"[INFO] Processing clip {clip_uid} (video_uid: {video_uid}) [{start_sec}-{end_sec}s]")
            video_start_time = time.time()
            caption_times = clip_to_caption_times.get(clip_uid, [])
            clip_length = end_sec - start_sec
            num_frames = int(math.floor(clip_length / 10))
            if num_frames < 1:
                num_frames = 1  # Always process at least one frame

            if not caption_times:
                print(f"[WARNING] No captioned timestamps for clip {clip_uid}, skipping.")
                continue

            # Only use caption_times within the clip interval
            valid_times = [t for t in caption_times if start_sec <= t <= end_sec]
            if not valid_times:
                print(f"[WARNING] No valid captioned timestamps within interval for clip {clip_uid}, skipping.")
                continue

            if len(valid_times) <= num_frames:
                selected_times = sorted(valid_times)
            else:
                selected_times = sorted(random.sample(valid_times, num_frames))

            frame_paths = extract_frames_at_timestamps(video_path, frames_dir, selected_times)

            frame_captions = []
            for frame_path, sec in frame_paths:
                img_b64 = encode_image_base64(frame_path)
                caption, frame_time = caption_frame(img_b64, clip_uid)
                if caption:
                    frame_captions.append({"frame": os.path.basename(frame_path), "second": sec, "caption": caption, "caption_time_sec": frame_time})
                    print(f"[INFO] Captioned frame at {sec}s in {frame_time:.2f} seconds.")
                else:
                    print(f"[WARNING] Failed to caption frame at {sec}s.")
                os.remove(frame_path)
            video_elapsed = time.time() - video_start_time
            clip_result = {
                "clip_uid": clip_uid,
                "video_uid": video_uid,
                "clip_start_sec": start_sec,
                "clip_end_sec": end_sec,
                "frame_captions": frame_captions,
                "clip_processing_time_sec": video_elapsed
            }
            results.append(clip_result)
            processed += 1
            print(f"[INFO] Processed {len(frame_captions)} frames for {clip_uid} in {video_elapsed:.2f} seconds. ({processed}/{total_clips})")
        # Save after every video
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        video_count += 1
    print(f"[INFO] Saved results to {output_json}")

if __name__ == "__main__":
    main() 