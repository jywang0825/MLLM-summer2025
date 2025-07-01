# ollama_qwen_caption_benchmark.py
import json
import sqlite3
import requests
import os
from datetime import datetime
import cv2
import base64
import time

# Database setup
def setup_database():
    pass  # No longer needed

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_frames_every_n_seconds(video_path, output_dir, interval_sec=5): # time between frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frame_paths = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    times = [int(t) for t in range(0, int(duration), interval_sec)]
    for idx, sec in enumerate(times):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{idx:02d}_{sec}s.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append((frame_path, sec))
    cap.release()
    return frame_paths

def caption_frame(img_b64, video_name):
    url = "http://localhost:11434/api/generate"
    prompt = f"Describe the main activity in this egocentric video frame from {video_name}."
    payload = {
        "model": "llava-phi3:latest",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    start = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    if response.status_code == 200:
        return response.json().get("response"), elapsed
    return None, elapsed

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

# Main caption generation and JSON output
def main():
    clips_directory = os.path.expanduser('~/remote_ego4d/v2/clips')
    output_json = 'llava_frame_captions.json'
    possible_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    results = []
    video_files = [f for f in os.listdir(clips_directory)
                   if os.path.isfile(os.path.join(clips_directory, f)) and os.path.splitext(f)[1].lower() in possible_extensions]
    video_files = video_files[:3]  # Limit for benchmarking
    frames_dir = 'llava_extracted_frames'
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Found {len(video_files)} video files to process")
    for i, video_filename in enumerate(video_files):
        video_path = os.path.join(clips_directory, video_filename)
        print(f"Processing {i+1}/{len(video_files)}: {video_filename}")
        video_start_time = time.time()
        frame_paths = extract_frames_every_n_seconds(video_path, frames_dir, interval_sec=10)
        frame_captions = []
        for frame_path, sec in frame_paths:
            img_b64 = encode_image_base64(frame_path)
            caption, frame_time = caption_frame(img_b64, video_filename)
            if caption:
                frame_captions.append({"frame": os.path.basename(frame_path), "second": sec, "caption": caption, "caption_time_sec": frame_time})
                print(f"Captioned frame at {sec}s in {frame_time:.2f} seconds.")
            else:
                print(f"Failed to caption frame at {sec}s.")
            os.remove(frame_path)
        video_elapsed = time.time() - video_start_time
        results.append({
            "video_filename": video_filename,
            "frame_captions": frame_captions,
            "video_processing_time_sec": video_elapsed
        })
        print(f"Processed {len(frame_captions)} frames for {video_filename} in {video_elapsed:.2f} seconds.")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_json}")

if __name__ == "__main__":
    main() 