#!/usr/bin/env python3
"""
Example Usage: Combining Uniform Frame Captioning with Frame Annotations
This script demonstrates how to use the frame annotations extractor with your existing captioning script.
"""

import json
import os
from extract_ego4d_frame_annotations import main as extract_annotations
from ego4d_aks_caption_uniform_intern2.5_8b import main as generate_captions

def run_complete_pipeline():
    """Run the complete pipeline: extract annotations + generate captions."""
    
    print("Ego4D Complete Pipeline: Annotations + Captions")
    print("=" * 60)
    
    # Step 1: Extract frame annotations
    print("\nStep 1: Extracting frame annotations...")
    print("-" * 40)
    
    # You can run this as a separate script or import the functions
    # For this example, we'll assume you've already run the extractor
    annotations_file = "ego4d_uniform_frame_annotations.json"
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file {annotations_file} not found.")
        print("Please run the extractor first:")
        print("python extract_ego4d_frame_annotations.py --max_videos 5")
        return
    
    # Load the extracted annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    print(f"Loaded annotations for {len(annotations_data)} videos")
    
    # Step 2: Generate captions (using your existing script)
    print("\nStep 2: Generating captions...")
    print("-" * 40)
    
    captions_file = "ego4d_uniform_captions_intern2.5_8b.json"
    
    if not os.path.exists(captions_file):
        print(f"Captions file {captions_file} not found.")
        print("Please run the captioning script first:")
        print("python ego4d_aks_caption_uniform_intern2.5_8b.py")
        return
    
    # Load the generated captions
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    print(f"Loaded captions for {len(captions_data)} videos")
    
    # Step 3: Combine annotations and captions
    print("\nStep 3: Combining annotations and captions...")
    print("-" * 40)
    
    # Create a mapping from video_uid to captions
    captions_map = {item['video_uid']: item for item in captions_data}
    
    # Combine the data
    combined_results = []
    
    for annotation_item in annotations_data:
        video_uid = annotation_item['video_uid']
        
        if video_uid in captions_map:
            caption_item = captions_map[video_uid]
            
            # Create combined result
            combined_result = {
                'video_uid': video_uid,
                'clip_uid': annotation_item['clip_uid'],
                'video_path': annotation_item['video_path'],
                'original_summary': annotation_item['original_summary'],
                'uniform_sampled_frames': annotation_item['uniform_sampled_frames'],
                'frame_indices': annotation_item['frame_indices'],
                'frame_timestamps': annotation_item['frame_timestamps'],
                'frame_captions': caption_item.get('frame_captions', []),
                'generated_summary': caption_item.get('generated_summary', ''),
                'frame_annotations': annotation_item['frame_annotations'],
                'frame_annotation_map': annotation_item['frame_annotation_map'],
                'total_annotations': annotation_item['total_annotations'],
                'timestamp': annotation_item['timestamp']
            }
            
            combined_results.append(combined_result)
            print(f"Combined data for video: {video_uid}")
        else:
            print(f"No captions found for video: {video_uid}")
    
    # Save combined results
    combined_file = "ego4d_combined_annotations_captions.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to: {combined_file}")
    print(f"Total combined videos: {len(combined_results)}")
    
    # Step 4: Analyze the results
    print("\nStep 4: Analysis...")
    print("-" * 40)
    
    if combined_results:
        total_frames = sum(r['uniform_sampled_frames'] for r in combined_results)
        total_annotations = sum(r['total_annotations'] for r in combined_results)
        total_captions = sum(len(r['frame_captions']) for r in combined_results)
        
        print(f"Total frames processed: {total_frames}")
        print(f"Total annotations found: {total_annotations}")
        print(f"Total captions generated: {total_captions}")
        print(f"Average annotations per video: {total_annotations / len(combined_results):.2f}")
        print(f"Average captions per video: {total_captions / len(combined_results):.2f}")
        
        # Show example of a frame with both annotation and caption
        for result in combined_results[:3]:  # Show first 3 videos
            video_uid = result['video_uid']
            print(f"\nExample for video {video_uid}:")
            
            # Find frames with both annotations and captions
            for frame_idx in range(min(3, result['uniform_sampled_frames'])):  # Show first 3 frames
                timestamp = result['frame_timestamps'][frame_idx]
                caption = result['frame_captions'][frame_idx] if frame_idx < len(result['frame_captions']) else "No caption"
                
                # Get annotations for this frame
                frame_annotations = result['frame_annotation_map'].get(frame_idx, [])
                
                print(f"  Frame {frame_idx} ({timestamp:.2f}s):")
                print(f"    Caption: {caption[:100]}...")
                if frame_annotations:
                    for ann in frame_annotations:
                        if ann['annotation_type'] == 'moments':
                            print(f"    Moments: {ann['label']} (primary: {ann['primary']})")
                        elif ann['annotation_type'] == 'narration':
                            print(f"    Narration: {ann['narration_text'][:50]}...")
                else:
                    print(f"    Annotations: None")

def show_usage_examples():
    """Show usage examples for the scripts."""
    
    print("Usage Examples:")
    print("=" * 60)
    
    print("\n1. Extract frame annotations (run this first):")
    print("   python extract_ego4d_frame_annotations.py --max_videos 5")
    print("   python extract_ego4d_frame_annotations.py --num_frames 16 --max_videos 10")
    
    print("\n2. Generate captions (run this second):")
    print("   python ego4d_aks_caption_uniform_intern2.5_8b.py")
    
    print("\n3. Run complete pipeline:")
    print("   python example_usage.py")
    
    print("\n4. Custom annotation extraction:")
    print("   python extract_ego4d_frame_annotations.py \\")
    print("     --moments_path ../remote_ego4d/v2/annotations/moments_train.json \\")
    print("     --narration_path ../remote_ego4d/v2/annotations/narration.json \\")
    print("     --output_file ego4d_train_annotations.json \\")
    print("     --num_frames 64 --max_videos 20")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage_examples()
    else:
        run_complete_pipeline() 