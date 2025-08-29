#!/usr/bin/env python3
"""
Test AKS captioning on videos that were missing from the original AKS processing
"""
import json
import os

def get_missing_video_uids():
    """Get the list of video UIDs that were missing from AKS processing."""
    missing_uids = [
        "000cd456-ff8d-499b-b0c1-4acead128a8b",
        "0aec2b0c-18e1-453b-a6ed-e484698d41df",
        "0de39e75-fb19-47d4-818d-fff874b05ab9",
        "3e92e0d2-2468-47f1-bbdf-1ebe1eb736c8",
        "5cab806a-3794-4578-94f2-33c3093151d1",
        "5fce50cf-ebc5-4c09-8704-20c48fe42c7e",
        "79c62f24-488e-4a69-8220-3b20cb4bf72b",
        "86ebad37-486c-4687-b50e-67627a4be7a2",
        "8717d09f-391c-4203-8e6c-7fa1f035cdda",
        "9239bf0f-db8c-4378-866d-15f302da08b5",
        "94cdabf3-c078-4ad4-a3a1-c42c8fc3f4ad",
        "b9eed644-56a9-4a0b-b1a3-5cc0d6688297",
        "bedaa131-deae-4279-9fd4-5b6ab552644c",
        "c0f702e1-d2e3-4277-8981-7177d106174a",
        "c1e648e-2457-4f99-a25d-21701bed69ec",
        "d340e569-12d3-42ef-a56b-a9a25c37ef95",
        "f681f510-cd33-48e3-bc10-4a8f2a518495",
        "fd06094f-8e2f-414c-9f66-3c99cd14a151"
    ]
    return missing_uids

def find_video_path(video_uid, clips_manifest_path="../remote_ego4d/v2/clips/manifest.csv"):
    """Find the video file path for a given video UID from the manifest."""
    try:
        import pandas as pd
        df = pd.read_csv(clips_manifest_path)
        # Look for the video UID in the exported_clip_uid column
        match = df[df['exported_clip_uid'] == video_uid]
        if not match.empty:
            # Use the manifold_location path
            manifold_path = match.iloc[0]['manifold_location']
            # Convert manifold path to local path
            local_path = manifold_path.replace('manifold://ego4d_fair/tree/exported_clips/', '../remote_ego4d/v2/clips/')
            return local_path
        else:
            print(f"Video UID {video_uid} not found in manifest")
            return None
    except Exception as e:
        print(f"Error finding video path for {video_uid}: {e}")
        return None

def check_video_files():
    """Check if the missing videos exist in the clips directory."""
    missing_uids = get_missing_video_uids()
    
    print("Checking missing videos...")
    print("=" * 50)
    
    found_videos = []
    missing_videos = []
    
    for video_uid in missing_uids:
        video_path = find_video_path(video_uid)
        if video_path and os.path.exists(video_path):
            print(f"✅ Found: {video_uid}")
            print(f"   Path: {video_path}")
            found_videos.append(video_uid)
        else:
            print(f"❌ Missing: {video_uid}")
            missing_videos.append(video_uid)
        print()
    
    print("=" * 50)
    print(f"Found: {len(found_videos)} videos")
    print(f"Missing: {len(missing_videos)} videos")
    
    return found_videos, missing_videos

def create_test_aks_input(found_videos):
    """Create a test input file for AKS processing with just the found videos."""
    # Load NLQ validation data to get clip info
    with open("../remote_ego4d/v2/annotations/nlq_val.json", 'r') as f:
        nlq_data = json.load(f)
    
    # Load summaries
    with open("../v1/nlq_val_summaries.json", 'r') as f:
        summaries_data = json.load(f)
    
    summary_map = {item['video_uid']: item['summary'] for item in summaries_data}
    
    # Create AKS-style input
    aks_input = []
    for video in nlq_data['videos']:
        if video['video_uid'] in found_videos:
            for clip in video['clips']:
                aks_input.append({
                    'video_uid': video['video_uid'],
                    'clip_uid': clip['clip_uid'],
                    'summary': summary_map.get(video['video_uid'], 'No summary available'),
                    'video_path': find_video_path(clip['clip_uid'])
                })
    
    # Save test input
    with open('test_missing_videos_aks_input.json', 'w') as f:
        json.dump(aks_input, f, indent=2)
    
    print(f"Created test input with {len(aks_input)} clips")
    return aks_input

if __name__ == "__main__":
    found_videos, missing_videos = check_video_files()
    
    if found_videos:
        print("\nCreating test AKS input file...")
        test_input = create_test_aks_input(found_videos)
        print(f"Test input saved to: test_missing_videos_aks_input.json")
        
        print("\nYou can now run the AKS script on these videos using:")
        print("python ../v1/ego4d_aks_caption_internvl_lmdeploy.py test_missing_videos_aks_input.json")
    else:
        print("No missing videos found in clips directory!") 