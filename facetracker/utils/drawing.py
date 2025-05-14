"""
Drawing utilities for visualizing face tracking and analysis results.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import mediapipe as mp # Added for pose drawing
from mediapipe.framework.formats import landmark_pb2 # Added for pose drawing
from tqdm import tqdm # Added for progress bar

# Helper to get a consistent color for an ID
def _get_color_for_id(obj_id: int) -> Tuple[int, int, int]:
    """Assigns a unique, visually distinct color for a given ID."""
    # Simple hashing to get somewhat consistent colors
    val = obj_id * 10
    r = (val * 23) % 255
    g = (val * 41) % 255
    b = (val * 67) % 255
    # Ensure color is not too dark
    r = max(50, r)
    g = max(50, g)
    b = max(50, b)
    return (r, g, b)

# Helper to convert landmark list to NormalizedLandmarkList proto
# (as seen in scripts/visualize_clusters.py and facetracker/pose_estimator.py)
def _to_landmark_list_proto(landmarks_data: List[List[float]], is_pose: bool) -> landmark_pb2.NormalizedLandmarkList:
    """Converts a list of landmark data [x,y,z,visibility] to NormalizedLandmarkList proto."""
    landmark_list_proto = landmark_pb2.NormalizedLandmarkList()
    for lm_data in landmarks_data:
        landmark = landmark_list_proto.landmark.add()
        landmark.x = lm_data[0]
        landmark.y = lm_data[1]
        if len(lm_data) > 2: # z might not always be present
            landmark.z = lm_data[2]
        if len(lm_data) > 3: # visibility/presence
             # For pose, visibility is common. For face mesh, sometimes presence is used.
            landmark.visibility = lm_data[3] 
    return landmark_list_proto

# TODO: Define drawing functions like draw_face_bbox_with_id, draw_cluster_id_on_video, etc.
# pass # Remove existing pass

def draw_face_bbox_with_id(
    video_path: str,
    tracked_data_by_scene: Dict[str, List[Dict[str, Any]]],
    output_video_path: str,
    draw_pose: bool = False,
    font_scale: float = 0.7,
    thickness: int = 2
) -> None:
    """
    Draws face bounding boxes with track IDs on a video.

    Args:
        video_path (str): Path to the input video.
        tracked_data_by_scene (Dict[str, List[Dict[str, Any]]]): 
            Tracked data. Example: {"scene_0": [{"frame": 0, "id": 1, "bbox": [x1,y1,x2,y2], "full_body_pose_landmarks": [[x,y,z,vis],...]}, ...]}
        output_video_path (str): Path to save the annotated video.
        draw_pose (bool): If True, attempts to draw associated pose landmarks.
        font_scale (float): Font scale for text.
        thickness (int): Thickness for bounding boxes and text.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    # Reorganize data for easy lookup by frame index
    data_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for scene_id, tracks_in_scene in tracked_data_by_scene.items():
        for track_item in tracks_in_scene:
            frame_idx = track_item.get("frame")
            if frame_idx is not None:
                if frame_idx not in data_by_frame:
                    data_by_frame[frame_idx] = []
                data_by_frame[frame_idx].append(track_item)
    
    # Initialize MediaPipe drawing utilities if drawing pose
    mp_drawing = None
    mp_pose_connections = None
    if draw_pose:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose_connections = mp.solutions.pose.POSE_CONNECTIONS


    for frame_num in tqdm(range(total_frames), desc=f"Drawing Track IDs on {output_video_path}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num}. Stopping.")
            break

        tracks_in_current_frame = data_by_frame.get(frame_num, [])
        
        for track_info in tracks_in_current_frame:
            track_id = track_info.get("id")
            bbox = track_info.get("bbox") # Expected: [x1, y1, x2, y2]

            if track_id is None or bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            color = _get_color_for_id(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"ID: {track_id}"
            
            # Calculate text size and position
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y_pos = max(y1, label_height + 10) # Ensure label is not above the frame
            
            # Draw filled rectangle for label background
            cv2.rectangle(frame, 
                          (x1, label_y_pos - label_height - baseline), 
                          (x1 + label_width, label_y_pos - baseline + 5), # Adjusted for slight padding
                          color, 
                          cv2.FILLED)
            # Draw label text (white text for better contrast on colored background)
            cv2.putText(frame, label, (x1, label_y_pos - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Draw pose landmarks if requested and available
            if draw_pose and mp_drawing and "full_body_pose_landmarks" in track_info:
                pose_landmarks_data = track_info.get("full_body_pose_landmarks")
                if pose_landmarks_data and isinstance(pose_landmarks_data, list) and len(pose_landmarks_data) > 0:
                    # Ensure data is in correct format (list of lists/tuples [x,y,z,vis])
                    # The `PersonAssociator` stores it as [[lm.x, lm.y, lm.z, lm.visibility or lm.presence]]
                    # which matches the input for _to_landmark_list_proto
                    try:
                        pose_landmarks_proto = _to_landmark_list_proto(pose_landmarks_data, is_pose=True)
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=pose_landmarks_proto,
                            connections=mp_pose_connections,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1), # Use track color for landmarks
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1) # Use track color for connections
                        )
                    except Exception as e:
                        print(f"Warning: Could not draw pose for track {track_id} in frame {frame_num}. Error: {e}")


        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Finished drawing track IDs. Output saved to: {output_video_path}")

def draw_cluster_id_on_video(
    video_path: str,
    clustered_data_by_scene: Dict[str, List[Dict[str, Any]]],
    output_video_path: str,
    draw_pose: bool = False, # Added optional draw_pose argument
    font_scale: float = 0.7,
    thickness: int = 2
) -> None:
    """
    Draws cluster IDs on faces in a video.

    Args:
        video_path (str): Path to the input video.
        clustered_data_by_scene (Dict[str, List[Dict[str, Any]]]): 
            Clustered data. Assumes structure similar to tracked_data but with cluster_id.
            Example: {"scene_0": [{"frame": 0, "id": 1, "cluster_id": 3, "bbox": [x1,y1,x2,y2], "full_body_pose_landmarks": [...]}, ...]}
        output_video_path (str): Path to save the annotated video.
        draw_pose (bool): If True, attempts to draw associated pose landmarks.
        font_scale (float): Font scale for text.
        thickness (int): Thickness for bounding boxes and text.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    # Reorganize data for easy lookup by frame index
    data_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    # The structure of clustered_data_by_scene is usually:
    # { scene_id: [ {unique_track_id: "scene_X_track_Y", cluster_id: Z, frames: [ {frame_idx: F, bbox: B, ...}, ... ] }, ... ] }
    # Or it could be flatter like tracked_data_by_scene if FaceClusterer.cluster_embeddings output is just an augmented version of it.
    # Based on current run_pipeline.py, clustered_data_by_scene is the output of FaceClusterer.cluster_embeddings
    # which takes tracked_data_with_embeddings_by_scene. 
    # Let's assume it's like tracked_data: Dict[scene_id, List[track_dicts_with_cluster_id]]
    # where track_dicts_with_cluster_id has 'frame', 'bbox', 'cluster_id', 'full_body_pose_landmarks', etc.

    for scene_id, items_in_scene in clustered_data_by_scene.items():
        for item in items_in_scene: # item is a dict representing one instance of a face in a frame
            frame_idx = item.get("frame") # This is critical for frame-based processing
            if frame_idx is not None:
                if frame_idx not in data_by_frame:
                    data_by_frame[frame_idx] = []
                data_by_frame[frame_idx].append(item) # item already contains cluster_id, bbox etc.
            # else: 
                # print(f"Warning: Item in clustered_data_by_scene missing 'frame': {item}")
                # This might happen if the data structure is more nested like the commented example above.
                # If so, this data preparation step needs adjustment.
                # For now, proceeding with the assumption that `item` is a per-frame detection with `cluster_id`.

    # Initialize MediaPipe drawing utilities if drawing pose
    mp_drawing = None
    mp_pose_connections = None
    if draw_pose:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    for frame_num in tqdm(range(total_frames), desc=f"Drawing Cluster IDs on {output_video_path}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num}. Stopping.")
            break

        items_in_current_frame = data_by_frame.get(frame_num, [])
        
        for item_info in items_in_current_frame:
            cluster_id = item_info.get("cluster_id") # Key change here
            bbox = item_info.get("bbox")

            if cluster_id is None or bbox is None: # cluster_id can be 0, so check for None explicitly
                # print(f"Skipping item due to missing cluster_id or bbox in frame {frame_num}: {item_info}")
                continue

            x1, y1, x2, y2 = map(int, bbox)
            color = _get_color_for_id(cluster_id) # Color by cluster_id

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"Cluster: {cluster_id}" # Label change here
            
            # Calculate text size and position
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y_pos = max(y1, label_height + 10)
            
            cv2.rectangle(frame, 
                          (x1, label_y_pos - label_height - baseline), 
                          (x1 + label_width, label_y_pos - baseline + 5), 
                          color, 
                          cv2.FILLED)
            cv2.putText(frame, label, (x1, label_y_pos - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Draw pose landmarks if requested and available
            if draw_pose and mp_drawing and "full_body_pose_landmarks" in item_info:
                pose_landmarks_data = item_info.get("full_body_pose_landmarks")
                if pose_landmarks_data and isinstance(pose_landmarks_data, list) and len(pose_landmarks_data) > 0:
                    try:
                        pose_landmarks_proto = _to_landmark_list_proto(pose_landmarks_data, is_pose=True)
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=pose_landmarks_proto,
                            connections=mp_pose_connections,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1), # Color by cluster_id
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1) # Color by cluster_id
                        )
                    except Exception as e:
                        print(f"Warning: Could not draw pose for cluster {cluster_id} in frame {frame_num}. Error: {e}")

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Finished drawing cluster IDs. Output saved to: {output_video_path}")

def draw_selected_frames_on_video(
    input_video_path: str, # Changed from video_path to avoid conflict if it's modified
    selected_frames_data: Dict[str, List[Dict[str, Any]]], # Data structure from FrameSelector.select_best_frames_per_cluster
    output_video_path: str,
    marker_color: Tuple[int, int, int] = (0, 255, 255), # Yellow for markers
    marker_radius: int = 15, # Radius of the marker circle
    marker_thickness: int = -1 # Filled circle
) -> None:
    """
    Marks selected frames on a video (e.g., by drawing a small marker).

    Args:
        input_video_path (str): Path to the video on which to draw markers.
        selected_frames_data (Dict[str, List[Dict[str, Any]]]): 
            Output from FrameSelector.select_best_frames_per_cluster.
            Example: {"scene_0": [{"unique_track_id": "scene_0_track_1", "top_frames": [{"frame_idx": 100, ...}, ...]}, ...]}
        output_video_path (str): Path to save the annotated video.
        marker_color (Tuple[int,int,int]): Color for the markers.
        marker_radius (int): Radius of the marker circle.
        marker_thickness (int): Thickness for the marker (e.g., -1 for filled).
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path} for marking selected frames.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    # Extract all unique selected frame indices
    selected_frame_indices = set()
    for scene_id, tracks_in_scene in selected_frames_data.items():
        for track_data in tracks_in_scene:
            if "top_frames" in track_data:
                for frame_info in track_data["top_frames"]:
                    if "frame_idx" in frame_info:
                        selected_frame_indices.add(frame_info["frame_idx"])
    
    if not selected_frame_indices:
        print("Warning: No selected frames found in selected_frames_data. Output video will be a copy of the input.")

    for frame_num in tqdm(range(total_frames), desc=f"Marking Selected Frames on {output_video_path}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num} from {input_video_path}. Stopping.")
            break

        if frame_num in selected_frame_indices:
            # Draw a marker (e.g., a circle in the top-right corner)
            marker_position = (frame_width - marker_radius - 10, marker_radius + 10) # 10px padding
            cv2.circle(frame, marker_position, marker_radius, marker_color, marker_thickness)
        
        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Finished marking selected frames. Output saved to: {output_video_path}") 