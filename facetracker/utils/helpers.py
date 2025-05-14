"""
Helper functions for the facetracker package.
"""
import os
import cv2
import csv
from typing import Dict, List, Any
from tqdm import tqdm

def save_scene_clips(
    original_video_path: str, 
    data_by_scene: Dict[str, List[Dict[str, Any]]], 
    output_clips_dir: str
) -> None:
    """
    Saves video clips for each scene based on the frame ranges found in data_by_scene.

    Args:
        original_video_path (str): Path to the full original video file.
        data_by_scene (Dict[str, List[Dict[str, Any]]]): 
            Data structured by scene. Keys are scene_ids (e.g., "scene_0").
            Each value is a list of dictionaries, where each dictionary should 
            at least contain a "frame" key indicating the frame number.
            Example: {"scene_0": [{"frame": 0, ...}, {"frame": 10, ...}, ..., {"frame": 100, ...}]}
        output_clips_dir (str): Directory where the scene clips will be saved.
    """
    os.makedirs(output_clips_dir, exist_ok=True)
    print(f"Saving scene clips to: {output_clips_dir}")

    cap_main = cv2.VideoCapture(original_video_path)
    if not cap_main.isOpened():
        print(f"Error: Could not open original video {original_video_path} for saving clips.")
        return

    main_fps = cap_main.get(cv2.CAP_PROP_FPS)
    main_width = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    main_height = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_main_frames = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_main.release() # Release now, will reopen per clip or read all frames once

    if not data_by_scene:
        print("No scene data provided to save_scene_clips. Nothing to do.")
        return

    for scene_id, items_in_scene in data_by_scene.items():
        if not items_in_scene:
            print(f"Scene {scene_id} has no items, skipping clip generation.")
            continue

        frame_numbers = sorted([item["frame"] for item in items_in_scene if "frame" in item])
        if not frame_numbers:
            print(f"Scene {scene_id} has no frame numbers associated with its items. Skipping.")
            continue
        
        min_frame = min(frame_numbers)
        max_frame = max(frame_numbers)

        scene_clip_path = os.path.join(output_clips_dir, f"{scene_id}_clip.mp4")
        print(f"Generating clip for {scene_id} (frames {min_frame}-{max_frame}) -> {scene_clip_path}")

        # Re-open main video capture for reading frames for this specific clip
        # This is less efficient than reading all frames once if memory allows, but safer for long videos.
        cap_clip_read = cv2.VideoCapture(original_video_path)
        if not cap_clip_read.isOpened():
            print(f"Error: Could not re-open video {original_video_path} to extract clip for scene {scene_id}.")
            continue
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_clip_writer = cv2.VideoWriter(scene_clip_path, fourcc, main_fps, (main_width, main_height))
        if not out_clip_writer.isOpened():
            print(f"Error: Could not open video writer for {scene_clip_path}")
            cap_clip_read.release()
            continue

        current_frame_idx = 0
        frames_written_for_clip = 0
        
        # Set starting position for reading
        cap_clip_read.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        current_frame_idx = min_frame

        for _ in tqdm(range(min_frame, max_frame + 1), desc=f"Writing {scene_id}"):
            if current_frame_idx > max_frame:
                break # Should not happen if loop is correct, but as a safeguard
            
            ret, frame = cap_clip_read.read()
            if not ret:
                print(f"Warning: Could not read frame {current_frame_idx} while generating clip for {scene_id}. Clip might be shorter.")
                break
            
            out_clip_writer.write(frame)
            frames_written_for_clip += 1
            current_frame_idx += 1

        print(f"Scene {scene_id}: Wrote {frames_written_for_clip} frames to {scene_clip_path}.")
        out_clip_writer.release()
        cap_clip_read.release()

    print("Finished processing scene clips.")


def save_cluster_data_to_csv(
    clustered_data_by_scene: Dict[str, List[Dict[str, Any]]], 
    csv_output_path: str
) -> None:
    """
    Saves a summary of cluster data to a CSV file.
    Each row could represent a unique face track instance within a cluster,
    including its original track ID, assigned cluster ID, and frame numbers.

    Args:
        clustered_data_by_scene (Dict[str, List[Dict[str, Any]]]): 
            Clustered data. Example:
            {"scene_0": [
                {"id": 1, "frame": 0, "bbox": [...], "cluster_id": 3, ...}, 
                {"id": 1, "frame": 1, "bbox": [...], "cluster_id": 3, ...}, 
                {"id": 2, "frame": 5, "bbox": [...], "cluster_id": 7, ...}
            ]}
            (This assumes a flat list of detections per scene, each with a cluster_id)
            Alternatively, if it's structured as FrameSelector output:
            {"scene_0": [
                {"unique_track_id": "scene_0_track_1", "cluster_id": 3, "top_frames": [...]}, 
                ...
            ]}
            The current implementation will assume the flatter structure as per draw_cluster_id_on_video.
        csv_output_path (str): Path to save the CSV file.
    """
    print(f"Saving cluster data summary to CSV: {csv_output_path}")
    fieldnames = [
        "scene_id", 
        "original_track_id", # from track_info["id"]
        "cluster_id", 
        "frame_index", 
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "confidence",
        "face_pose_iou" # Example of other data that might be useful
    ]

    rows_written = 0
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for scene_id, items_in_scene in clustered_data_by_scene.items():
            for item in items_in_scene: # Each item is a dictionary for a detected face instance
                bbox = item.get("bbox", [None,None,None,None])
                row_data = {
                    "scene_id": scene_id,
                    "original_track_id": item.get("id"),
                    "cluster_id": item.get("cluster_id"),
                    "frame_index": item.get("frame"),
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "confidence": item.get("confidence"),
                    "face_pose_iou": item.get("face_pose_iou")
                }
                writer.writerow(row_data)
                rows_written +=1
                
    print(f"Wrote {rows_written} rows to {csv_output_path}.") 