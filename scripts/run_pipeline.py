import os
import json
import cv2 # For getting image dimensions
import numpy as np # For NumpyEncoder if saving embeddings
from tqdm import tqdm # For overall progress if desired, though sub-modules have it
import matplotlib.pyplot as plt # Added for get_color
import argparse

# Assuming your package structure allows these imports when run from project root
# or that 'facetracker' is in PYTHONPATH.
# If running as a script, you might need to adjust sys.path or run as a module.
from facetracker.scene_detector import SceneDetector
from facetracker.face_detector import FaceDetector
from facetracker.face_tracker import FaceTracker, FrameSelector
from facetracker.face_cluster import FaceEmbedder, FaceClusterer

# Helper class for JSON serialization of NumPy arrays (like embeddings)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# --- Helper Function for Colors ---
def get_color(idx):
    """Gets a distinct color for a given ID."""
    cmap = plt.get_cmap('tab10')  # Using tab10 colormap
    return tuple(int(c * 255) for c in cmap(idx % 10)[:3]) # Get RGB, scale to 0-255

def run_full_pipeline(video_path: str, output_base_dir: str,
                      face_landmarker_model_path: str,
                      pose_landmarker_model_path: str):
    """
    Runs the full face tracking and clustering pipeline.
    """
    # Get video name without extension for the subfolder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    
    print(f"Starting pipeline for video: {video_path}")
    print(f"Output will be saved in: {output_dir}")

    # --- 0. Setup Output Directories ---
    os.makedirs(output_dir, exist_ok=True)
    scene_output_file = os.path.join(output_dir, "scene_cuts.csv")
    # FaceDetector's raw output can be very large, usually not saved unless debugging
    # face_detection_output_file = os.path.join(output_dir, "all_face_detections.json")
    frame_selector_crops_dir = os.path.join(output_dir, "selected_face_crops")
    os.makedirs(frame_selector_crops_dir, exist_ok=True)
    final_clustering_output_file = os.path.join(output_dir, "final_face_clusters.json")
    tracking_video_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
    
    # Optional: For saving intermediate results for debugging
    # intermediate_frame_selector_output_file = os.path.join(output_dir, "dbg_frame_selector_output.json")
    # intermediate_embedder_output_file = os.path.join(output_dir, "dbg_embedder_output.json")


    # --- 1. Scene Detection ---
    print("\nStep 1: Detecting scenes...")
    scene_detector = SceneDetector(video_path=video_path, min_scene_len=15) # min_scene_len is an example
    scene_detector.initialize_scene_manager()
    scene_detector.detect_scenes()
    scenes = scene_detector.shots # Retrieve the shots stored by detect_scenes

    if not scenes: # Handle case with no scenes detected, treat video as one scene
        print("No distinct scenes detected. Treating video as a single scene.")
        cap_temp = cv2.VideoCapture(video_path)
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_temp.get(cv2.CAP_PROP_FPS) if cap_temp.get(cv2.CAP_PROP_FPS) > 0 else 30.0
        cap_temp.release()
        if total_frames > 0:
            scenes = [(0, total_frames - 1, 0.0, (total_frames - 1) / fps)]
        else:
            print(f"Error: Video at {video_path} has no frames or could not be read.")
            return
    scene_detector.save_shots(scene_output_file)
    print(f"Detected {len(scenes)} scenes. Boundaries saved to {scene_output_file}")

    # --- 2. Face Detection (and Face Landmarks for entire video) ---
    print("\nStep 2: Detecting faces and face landmarks (this runs once)...")
    face_detector = FaceDetector(
        video_path=video_path,
        output_dir=output_dir, 
        face_landmarker_model_path=face_landmarker_model_path,
        # device='cuda' # Or 'cpu'. Default handles availability check.
        # Can add other MTCNN params here if needed, e.g.:
        face_min_confidence=0.95, # Adjust as needed
    )
    all_faces_data_by_frame_str = face_detector.detect_faces_in_video()
    # face_detector.save_results(face_detection_output_file, all_faces_data_by_frame_str) # Usually too large
    print("Face detection and landmark extraction complete for all frames.")

    # --- 3. Face Tracking (per scene) ---
    print("\nStep 3: Tracking faces within each scene...")
    tracked_data_by_scene = {} # Key: scene_id_str, Value: List[tracked_face_dict]

    cap_dims = cv2.VideoCapture(video_path)
    img_width = int(cap_dims.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap_dims.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_dims.release()
    if img_width == 0 or img_height == 0:
        print(f"Error: Could not get video dimensions from {video_path}.")
        if hasattr(face_detector, 'landmarker') and face_detector.landmarker: face_detector.landmarker.close()
        return
    img_size = (img_height, img_width)

    for i, scene_info in enumerate(scenes):
        scene_id_str = f"scene_{i}"
        start_frame, end_frame, _, _ = scene_info
        print(f"  Processing {scene_id_str}: frames {start_frame} to {end_frame}")

        face_tracker_instance = FaceTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        current_scene_tracked_data = []
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc=f"  Tracking {scene_id_str}", unit="frame", leave=False):
            frame_key = f"frame_{frame_idx}"
            detections_for_this_frame = all_faces_data_by_frame_str.get(frame_key, [])

            tracked_instances_in_frame = face_tracker_instance.track_faces(
                frame=frame_idx,
                face_data=detections_for_this_frame,
                img_size=img_size
            )
            current_scene_tracked_data.extend(tracked_instances_in_frame)
        
        tracked_data_by_scene[scene_id_str] = current_scene_tracked_data
        print(f"  Finished tracking for {scene_id_str}. Found {len(current_scene_tracked_data)} total tracked face instances in this scene.")
    
    if hasattr(face_detector, 'landmarker') and face_detector.landmarker:
        face_detector.landmarker.close() # Explicitly close FaceLandmarker
    del face_detector # Release resources

    # After tracking is complete, create visualization video
    print("\nCreating tracking visualization video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file for visualization: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(tracking_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Creating visualization video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find which scene this frame belongs to
        current_scene = None
        for scene_id, scene_data in tracked_data_by_scene.items():
            # Get frame detections for this scene
            frame_detections = [d for d in scene_data if d['frame'] == frame_idx]
            if frame_detections:
                current_scene = scene_id
                break

        if current_scene:
            # Draw bounding boxes and IDs for this frame
            for track in tracked_data_by_scene[current_scene]:
                if track['frame'] == frame_idx:
                    bbox = track['bbox']
                    track_id = track['id']
                    landmarks = track.get('landmarks')

                    x1, y1, x2, y2 = map(int, bbox)
                    color = get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if landmarks:
                        for landmark in landmarks:
                            lm_x = int(landmark[0] * width)
                            lm_y = int(landmark[1] * height)
                            cv2.circle(frame, (lm_x, lm_y), 1, (0, 255, 0), -1)

        out_video.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out_video.release()
    pbar.close()
    print(f"Tracking visualization video saved to: {tracking_video_path}")

    # --- 4. Frame Selection (and Pose Estimation) ---
    print("\nStep 4: Selecting top frames per track and performing pose estimation...")
    frame_selector = FrameSelector(
        video_file=video_path,
        output_dir=frame_selector_crops_dir,
        save_images=True, # Set to False if you don't need face crops saved
        pose_model_asset_path=pose_landmarker_model_path,
        top_n=5, # Example: select top 5 frames per track
        face_to_pose_iou_threshold=0.3 # Example
    )
    selected_frames_output = frame_selector.select_top_frames_per_face(tracked_data_by_scene)
    frame_selector.close() # Close PoseLandmarker
    # with open(intermediate_frame_selector_output_file, 'w') as f: json.dump(selected_frames_output, f, indent=4, cls=NumpyEncoder)
    print("Frame selection and pose estimation complete.")

    # --- 5. Face Embedding ---
    print("\nStep 5: Generating face embeddings...")
    face_embedder = FaceEmbedder() # Uses default InceptionResnetV1
    all_tracks_with_embeddings = face_embedder.get_face_embeddings(
        selected_frames_by_scene=selected_frames_output,
        image_dir=frame_selector_crops_dir
    )
    # with open(intermediate_embedder_output_file, 'w') as f: json.dump(all_tracks_with_embeddings, f, indent=4, cls=NumpyEncoder)
    print("Face embedding generation complete.")
    if not all_tracks_with_embeddings:
        print("No embeddings were generated. Skipping clustering. Check previous steps for errors or empty selections.")
        return

    # --- 6. Face Clustering ---
    print("\nStep 6: Clustering faces...")
    face_clusterer = FaceClusterer(similarity_threshold=0.7, max_iterations=50) # Example params
    final_clusters = face_clusterer.cluster_faces(all_tracks_with_embeddings)
    
    print(f"Face clustering complete. Found {len(final_clusters)} unique face clusters.")

    # --- 7. Save Final Output ---
    print(f"\nSaving final clustering results to {final_clustering_output_file}...")
    with open(final_clustering_output_file, 'w') as f:
        json.dump(final_clusters, f, indent=4, cls=NumpyEncoder)
    
    print(f"Pipeline finished successfully! Final output: {final_clustering_output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full face tracking and clustering pipeline.")
    parser.add_argument("--video_path", type=str, required=True,
                      help="Path to the input video file.")
    parser.add_argument("--output_base_dir", type=str, required=True,
                      help="Base directory for all output files.")
    parser.add_argument("--face_landmarker_model_path", type=str, required=True,
                      help="Path to the MediaPipe face landmarker model.")
    parser.add_argument("--pose_landmarker_model_path", type=str, required=True,
                      help="Path to the MediaPipe pose landmarker model.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_base_dir, exist_ok=True)

    run_full_pipeline(
        video_path=args.video_path,
        output_base_dir=args.output_base_dir,
        face_landmarker_model_path=args.face_landmarker_model_path,
        pose_landmarker_model_path=args.pose_landmarker_model_path
    )