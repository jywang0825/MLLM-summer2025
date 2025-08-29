#!/usr/bin/env python3
"""
Quick script to check which videos have annotations and compare with NLQ videos
"""

import json
import os

def check_moments_annotations():
    """Check which videos in moments_val.json have annotations."""
    moments_path = "../remote_ego4d/v2/annotations/moments_val.json"
    
    print("Loading moments data...")
    with open(moments_path, 'r') as f:
        moments_data = json.load(f)
    
    videos_with_annotations = []
    total_videos = 0
    
    for video in moments_data.get('videos', []):
        total_videos += 1
        video_uid = video.get('video_uid')
        clips = video.get('clips', [])
        
        for clip in clips:
            clip_uid = clip.get('clip_uid')
            annotations = clip.get('annotations', [])
            
            if annotations:
                videos_with_annotations.append({
                    'video_uid': video_uid,
                    'clip_uid': clip_uid,
                    'num_annotations': len(annotations)
                })
    
    print(f"Total videos in moments_val.json: {total_videos}")
    print(f"Videos with annotations: {len(videos_with_annotations)}")
    
    # Show first few examples
    print("\nFirst 5 videos with annotations:")
    for i, video in enumerate(videos_with_annotations[:5]):
        print(f"  {i+1}. Video: {video['video_uid']}")
        print(f"     Clip: {video['clip_uid']}")
        print(f"     Annotations: {video['num_annotations']}")
    
    return videos_with_annotations

def check_nlq_videos():
    """Check which videos are in NLQ validation set."""
    nlq_path = "../remote_ego4d/v2/annotations/nlq_val.json"
    
    print("\nLoading NLQ validation data...")
    with open(nlq_path, 'r') as f:
        nlq_data = json.load(f)
    
    nlq_videos = []
    for video in nlq_data.get('videos', []):
        video_uid = video.get('video_uid')
        clips = video.get('clips', [])
        
        for clip in clips:
            clip_uid = clip.get('clip_uid')
            nlq_videos.append({
                'video_uid': video_uid,
                'clip_uid': clip_uid
            })
    
    print(f"Total videos in nlq_val.json: {len(nlq_videos)}")
    
    # Show first few examples
    print("\nFirst 5 NLQ videos:")
    for i, video in enumerate(nlq_videos[:5]):
        print(f"  {i+1}. Video: {video['video_uid']}")
        print(f"     Clip: {video['clip_uid']}")
    
    return nlq_videos

def find_overlap():
    """Find overlap between moments annotations and NLQ videos."""
    moments_videos = check_moments_annotations()
    nlq_videos = check_nlq_videos()
    
    # Create sets for comparison
    moments_clips = {v['clip_uid'] for v in moments_videos}
    nlq_clips = {v['clip_uid'] for v in nlq_videos}
    
    overlap = moments_clips.intersection(nlq_clips)
    
    print(f"\nOverlap analysis:")
    print(f"Moments clips: {len(moments_clips)}")
    print(f"NLQ clips: {len(nlq_clips)}")
    print(f"Overlap: {len(overlap)}")
    
    if overlap:
        print(f"\nClips with both moments annotations and NLQ data:")
        for clip_uid in list(overlap)[:10]:  # Show first 10
            print(f"  - {clip_uid}")
    else:
        print("\nNo overlap found! This explains why we're not finding annotations.")
        print("We need to use a different annotation file or approach.")

if __name__ == "__main__":
    find_overlap() 