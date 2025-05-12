# print("Attempting to import os...")
# import os
# print(f"Initial LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
# print("Attempting to import tensorflow...")
# import tensorflow as tf
# print(f"TF Version: {tf.__version__}")
# print(f"Physical GPUs: {tf.config.list_physical_devices('GPU')}")
# print("TensorFlow import successful.")

# print("Attempting to import cv2...")
# import cv2
# print("cv2 import successful")

# print("Attempting to import facetracker.face_detector...")
# from facetracker.face_detector import FaceDetector # This will bring in retina-face
# print("FaceDetector import successful")

# --- Original code below this line would be commented out or removed for this test ---
# ''' # This line should be removed
# Assuming running from project root or facetracker is in PYTHONPATH
import os # Already imported, but common to see it again
import argparse # Added for __main__
import cv2 # Already imported
import matplotlib.pyplot as plt # Added for get_color
from tqdm import tqdm # Added for pbar

from facetracker.scene_detector import SceneDetector
from facetracker.face_detector import FaceDetector
from facetracker.face_tracker import FaceTracker

# --- Helper Function for Colors ---
def get_color(idx):
    """Gets a distinct color for a given ID."""
    cmap = plt.get_cmap('tab10')  # Using tab10 colormap
    return tuple(int(c * 255) for c in cmap(idx % 10)[:3]) # Get RGB, scale to 0-255

def visualize_sort_tracking(video_path: str, output_video_path: str,
                             face_landmarker_model_path: str):
    """
    Visualizes SORT tracking on a video, drawing bounding boxes and IDs for tracked faces.
    Also detects scenes and re-initializes tracker for each scene.
    """
    # Initialize Scene Detector
    print("Step 1: Detecting scenes...")
   
    custom_scene_detector = SceneDetector(video_path=video_path)
    custom_scene_detector.initialize_scene_manager() # Make sure this is called
    custom_scene_detector.detect_scenes()
    
    cap_temp_fps_check = cv2.VideoCapture(video_path)
    fps_for_shot_list = cap_temp_fps_check.get(cv2.CAP_PROP_FPS)
    if fps_for_shot_list <= 0: # Handle potential error where FPS is 0 or negative
        print(f"Warning: Invalid FPS ({fps_for_shot_list}) detected for video {video_path} when getting shot list. Using default 30.0.")
        fps_for_shot_list = 30.0
    cap_temp_fps_check.release()

    # get_shot_list returns: List[Tuple[int, int, float, float]] 
    # (start_frame, end_frame, start_time, end_time)
    # The tracking loop below expects scenes as (start_frame, end_frame)
    raw_scenes_from_custom_detector = custom_scene_detector.get_shot_list(fps_for_shot_list)
    scenes = []
    if raw_scenes_from_custom_detector:
        for s_frame, e_frame, _, _ in raw_scenes_from_custom_detector:
            scenes.append((s_frame, e_frame))
    else: # Fallback if custom detector returns no scenes
        print("No scenes detected by custom SceneDetector; falling back to processing entire video.")
        cap_fallback = cv2.VideoCapture(video_path)
        total_frames_fallback = int(cap_fallback.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fallback.release()
        if total_frames_fallback > 0:
            scenes = [(0, total_frames_fallback)]
        else:
            print("Could not determine video duration for fallback. Processing may fail.")
            # scenes will remain empty

    print(f"Scenes to process (frame ranges): {scenes}")

    # Initialize Face Detector
    # This will also initialize RetinaFace, which might be slow if not on GPU
    print("Step 2: Detecting faces and face landmarks for all frames...")
    face_detector = FaceDetector(video_path=video_path, output_dir="placeholder_output", 
                                 face_landmarker_model_path=face_landmarker_model_path)
    
    all_scene_detections = face_detector.detect_faces_in_video() # Process entire video once
    face_detector.close() # Close detector resources

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames_video, desc="Processing Video and Tracking")

    print("Step 3: Tracking faces across scenes...")
    current_scene_idx = 0
    if not scenes: # Should not happen if previous check is done
        print("Error: Scene list is empty before tracking.")
        return
        
    start_frame_current_scene, end_frame_current_scene = scenes[current_scene_idx]
    
    # Initialize Tracker (will be re-initialized per scene)
    face_tracker = FaceTracker(max_age=30, min_hits=3, iou_threshold=0.3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Scene change detection and tracker reset
        if frame_idx >= end_frame_current_scene:
            current_scene_idx += 1
            if current_scene_idx < len(scenes):
                start_frame_current_scene, end_frame_current_scene = scenes[current_scene_idx]
                # print(f"New scene ({current_scene_idx+1}/{len(scenes)}) from frame {start_frame_current_scene} to {end_frame_current_scene}. Re-initializing tracker.")
                face_tracker = FaceTracker(max_age=30, min_hits=3, iou_threshold=0.3) # Re-initialize tracker
            else:
                # print("Last scene processed.")
                pass # Continue processing remaining frames if any, or break
        
        # Get pre-computed detections for the current frame
        current_frame_key = f"frame_{frame_idx}" # Ensure key format matches FaceDetector output
        detections_for_frame = all_scene_detections.get(current_frame_key, [])
        
        # Update tracker with detections
        # The FaceTracker.track_faces expects a list of dicts with 'bbox' and 'confidence'
        # It also expects landmarks if available, which should be in detections_for_frame from FaceDetector
        tracked_objects = face_tracker.track_faces(frame_idx, detections_for_frame, (height, width)) 

        # Draw bounding boxes and IDs
        for track in tracked_objects:
            bbox = track['bbox'] # [x1, y1, x2, y2]
            track_id = track['id']
            # confidence = track.get('confidence', 0) # Get confidence if available
            landmarks = track.get('landmarks') # Get landmarks if available

            x1, y1, x2, y2 = map(int, bbox)
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if landmarks:
                for landmark in landmarks:
                    # Assuming landmark is a tuple or list (x, y)
                    # If landmarks are normalized (0.0 to 1.0), multiply by frame dimensions
                    lm_x = int(landmark[0] * width) 
                    lm_y = int(landmark[1] * height)
                    # If landmarks are already in pixel coordinates, use them directly:
                    # lm_x = int(landmark[0]) # This was incorrect for normalized landmarks
                    # lm_y = int(landmark[1]) # This was incorrect for normalized landmarks
                    cv2.circle(frame, (lm_x, lm_y), 1, (0, 255, 0), -1) # Draw small green dots

        out_video.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out_video.release()
    pbar.close()
    cv2.destroyAllWindows()
    print(f"Processing complete. Visualized video saved to {output_video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SORT tracking on a video with scene detection.")
    parser.add_argument("--video_path", type=str, default="/orcd/data/satra/002/datasets/cneuromod/friends/stimuli/s1/friends_s01e01a.mkv", help="Path to the input video file.")
    parser.add_argument("--output_video_path", type=str, default="facetracker_pipeline_output/video_tracking.mp4", help="Path to save the output video.")
    parser.add_argument("--face_landmarker_model_path", type=str, default="assets/face_landmarker.task", help="Path to the MediaPipe face landmarker model.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

    visualize_sort_tracking(args.video_path, args.output_video_path, args.face_landmarker_model_path)
