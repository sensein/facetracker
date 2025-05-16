import os
import argparse
import time
import json
from tqdm import tqdm
import cv2
import mediapipe as mp

from facetracker.face_detector import FaceDetector
from facetracker.face_tracker import FaceTracker, FrameSelector
from facetracker.face_cluster import FaceEmbedder, FaceClusterer
from facetracker.mmpose_estimator import MMPoseEstimator
from facetracker.person_associator import PersonAssociator
from facetracker.utils.drawing import draw_face_bbox_with_id, draw_cluster_id_on_video, draw_selected_frames_on_video
from facetracker.utils.helpers import save_scene_clips, save_cluster_data_to_csv

def main(args):
    print(f"Starting character tracking pipeline for video: {args.video_file}")
    start_time = time.time()

    output_base_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.video_file), "output_" + os.path.splitext(os.path.basename(args.video_file))[0])
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Output will be saved to: {output_base_dir}")

    # --- Step 1: Face Detection ---
    print("\nStep 1: Detecting faces...")
    face_detector_instance = FaceDetector(
        model_path=args.face_detection_model_path,
        device=args.device,
        output_dir=os.path.join(output_base_dir, "face_detection_output"),
        save_annotated_frames=args.save_annotated_detection_frames,
        detection_threshold=args.face_detection_threshold,
        target_face_size_ratio=args.target_face_size_ratio,
        skip_frames=args.detection_skip_frames
    )
    all_faces_data_by_frame_str = face_detector_instance.detect_faces_in_video(args.video_file)
    
    detection_output_path = os.path.join(output_base_dir, "all_faces_detected_data.json")
    with open(detection_output_path, 'w') as f:
        json.dump(all_faces_data_by_frame_str, f, indent=4)
    print(f"Face detection complete. Detected faces in {len(all_faces_data_by_frame_str)} frames.")
    print(f"Raw detection data saved to: {detection_output_path}")
    
    if not any(all_faces_data_by_frame_str.values()):
        print("No faces detected in the video. Exiting pipeline.")
        return

    # --- Step 1.5: Pose Estimation and Face-Pose Association ---
    print("\nStep 1.5: Estimating Poses and Associating with Faces...")
    mmpose_estimator = MMPoseEstimator(
        model_type=args.mmpose_model_type,
        model_config_path=args.mmpose_config_path,
        checkpoint_path=args.mmpose_checkpoint_path,
        device=args.device
    )
    
    person_associator_instance = PersonAssociator(
        pose_estimator=mmpose_estimator,
        face_to_pose_iou_threshold=args.face_to_pose_iou_threshold,
        temporal_window=args.pose_temporal_window,
        min_pose_confidence=args.min_pose_confidence
    )
    
    all_augmented_data_by_frame_str = person_associator_instance.process_and_associate(
        all_faces_data_by_frame_str, 
        args.video_file
    )
    
    mmpose_estimator.close()
    print("Pose estimation and association complete.")
    
    augmented_data_output_path = os.path.join(output_base_dir, "all_augmented_detection_data.json")
    with open(augmented_data_output_path, 'w') as f:
        json.dump(all_augmented_data_by_frame_str, f, indent=4)
    print(f"Augmented detection data (with pose info) saved to: {augmented_data_output_path}")

    # --- Get Video Dimensions for FaceTracker ---
    cap_temp_for_dims = cv2.VideoCapture(args.video_file)
    video_width = 0
    video_height = 0
    if cap_temp_for_dims.isOpened():
        video_width = int(cap_temp_for_dims.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap_temp_for_dims.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp_for_dims.release()
    else:
        print(f"Error: Could not open video {args.video_file} to get dimensions for FaceTracker. Exiting.")
        return

    if video_width == 0 or video_height == 0:
        print(f"Error: Video dimensions ({video_width}x{video_height}) are invalid. Exiting.")
        return

    # --- Step 2: Face Tracking with Pose Information ---
    print("\nStep 2: Tracking faces across frames...")
    face_tracker_instance = FaceTracker(
        iou_threshold=args.tracking_iou_threshold, 
        max_lost_tracks=args.max_lost_tracks,
        min_track_length=args.min_track_length,
        use_pose_info=True  # Enable pose information for tracking
    )
    
    tracked_data_by_scene = face_tracker_instance.track_faces(
        all_augmented_data_by_frame_str,
        video_width,
        video_height
    )
    
    tracking_output_path = os.path.join(output_base_dir, "tracked_faces_by_scene.json")
    with open(tracking_output_path, 'w') as f:
        json.dump(tracked_data_by_scene, f, indent=4)
    print(f"Face tracking complete. Found {len(tracked_data_by_scene)} scenes/tracks.")
    print(f"Tracking data saved to: {tracking_output_path}")

    if not tracked_data_by_scene:
        print("No stable tracks found after face tracking. Exiting further processing.")
        return

    tracking_video_path = os.path.join(output_base_dir, "tracking_visualization.mp4")
    draw_face_bbox_with_id(
        args.video_file, 
        tracked_data_by_scene, 
        tracking_video_path, 
        draw_pose=True,  # Always draw pose for better visualization
        show_pose_track_id=True  # Show pose track IDs
    )
    print(f"Tracking visualization video saved to: {tracking_video_path}")

    # --- Step 3: Face Embedding and Clustering ---
    print("\nStep 3: Generating face embeddings and clustering...")
    face_embedder = FaceEmbedder(
        model_name=args.embedding_model_name, 
        device=args.device,
        batch_size=args.embedding_batch_size
    )
    
    tracked_data_with_embeddings_by_scene = face_embedder.embed_faces_in_tracks(
        args.video_file, 
        tracked_data_by_scene,
        os.path.join(output_base_dir, "face_embeddings_cache")
    )
    
    embeddings_output_path = os.path.join(output_base_dir, "tracked_faces_with_embeddings.json")
    with open(embeddings_output_path, 'w') as f:
        json.dump(tracked_data_with_embeddings_by_scene, f, indent=4)
    print(f"Face embeddings generated. Saved to: {embeddings_output_path}")

    face_clusterer = FaceClusterer(
        metric=args.clustering_metric, 
        threshold=args.clustering_threshold,
        min_cluster_size=args.min_cluster_size
    )
    
    clustered_data_by_scene = face_clusterer.cluster_embeddings(tracked_data_with_embeddings_by_scene)
    
    clustering_output_path = os.path.join(output_base_dir, "clustered_face_data.json")
    with open(clustering_output_path, 'w') as f:
        json.dump(clustered_data_by_scene, f, indent=4)
    print(f"Face clustering complete. Saved to: {clustering_output_path}")

    # --- Step 4: Frame Selection ---
    print("\nStep 4: Selecting best frames per cluster...")
    frame_selector = FrameSelector(
        video_file=args.video_file,
        top_n=args.frame_selection_top_n,
        output_dir=os.path.join(output_base_dir, "selected_frames_per_cluster"),
        save_images=args.save_selected_frames
    )
    
    selected_frames_data = frame_selector.select_best_frames_per_cluster(clustered_data_by_scene)

    selected_frames_output_path = os.path.join(output_base_dir, "selected_frames_data.json")
    with open(selected_frames_output_path, 'w') as f:
        json.dump(selected_frames_data, f, indent=4)
    print(f"Frame selection complete. Data saved to: {selected_frames_output_path}")
    if args.save_selected_frames:
        print(f"Selected frames (images) saved to: {frame_selector.output_dir}")

    # --- Step 5: Final Output Generation ---
    print("\nStep 5: Generating final outputs...")
    if args.save_scene_clips:
        save_scene_clips(args.video_file, clustered_data_by_scene, os.path.join(output_base_dir, "scene_clips"))
        print(f"Scene clips saved to: {os.path.join(output_base_dir, 'scene_clips')}")

    csv_output_path = os.path.join(output_base_dir, "face_clusters_summary.csv")
    save_cluster_data_to_csv(clustered_data_by_scene, csv_output_path)
    print(f"Cluster summary CSV saved to: {csv_output_path}")
    
    final_video_path = os.path.join(output_base_dir, "final_clustered_output.mp4")
    draw_cluster_id_on_video(
        args.video_file, 
        clustered_data_by_scene, 
        final_video_path, 
        draw_pose=True,
        show_pose_track_id=True
    )
    print(f"Final video with cluster IDs saved to: {final_video_path}")

    if args.draw_selected_frames_on_final_video and args.save_selected_frames:
        final_video_with_selection_markers_path = os.path.join(output_base_dir, "final_video_with_selection_markers.mp4")
        draw_selected_frames_on_video(final_video_path, selected_frames_data, final_video_with_selection_markers_path)
        print(f"Final video with selected frame markers saved to: {final_video_with_selection_markers_path}")

    total_time = time.time() - start_time
    print(f"\nPipeline finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Character Tracking, Clustering, and Frame Selection Pipeline.")
    parser.add_argument("--video_file", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save all outputs. Defaults to 'output_[video_name]' in the video's directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for models ('cuda' or 'cpu').")

    # Face Detection (Step 1)
    parser.add_argument("--face_detection_model_path", type=str, default=None, help="Path to a custom face detection model. If None, uses a default.")
    parser.add_argument("--save_annotated_detection_frames", action="store_true", help="Save frames with detected face bounding boxes.")
    parser.add_argument("--face_detection_threshold", type=float, default=0.8, help="Confidence threshold for face detection.")
    parser.add_argument("--target_face_size_ratio", type=float, default=0.05, help="Target minimum face size as a ratio of the frame's smaller dimension (0-1).")
    parser.add_argument("--detection_skip_frames", type=int, default=0, help="Number of frames to skip between detections (0 for no skip).")

    # Pose Estimation & Association (Step 1.5)
    parser.add_argument("--mmpose_model_type", type=str, default="hiera_l", help="Type of MMPose model to use.")
    parser.add_argument("--mmpose_config_path", type=str, required=True, help="Path to the MMPose model configuration file.")
    parser.add_argument("--mmpose_checkpoint_path", type=str, required=True, help="Path to the MMPose model checkpoint file.")
    parser.add_argument("--face_to_pose_iou_threshold", type=float, default=0.3, help="IoU threshold for associating a face with a pose-derived head bounding box.")
    parser.add_argument("--pose_temporal_window", type=int, default=5, help="Number of frames to look back for temporal consistency in pose tracking.")
    parser.add_argument("--min_pose_confidence", type=float, default=0.3, help="Minimum confidence threshold for pose keypoints.")
    
    # Face Tracking (Step 2)
    parser.add_argument("--tracking_iou_threshold", type=float, default=0.4, help="IoU threshold for matching detections to existing tracks.")
    parser.add_argument("--max_lost_tracks", type=int, default=10, help="Maximum number of consecutive frames a track can be lost.")
    parser.add_argument("--min_track_length", type=int, default=5, help="Minimum number of frames a track must exist to be considered valid.")

    # Face Embedding (Step 3)
    parser.add_argument("--embedding_model_name", type=str, default="facenet", help="Name of the face embedding model.")
    parser.add_argument("--embedding_batch_size", type=int, default=32, help="Batch size for face embedding generation.")
    
    # Clustering (Step 3)
    parser.add_argument("--clustering_metric", type=str, default="cosine", help="Distance metric for clustering.")
    parser.add_argument("--clustering_threshold", type=float, default=0.6, help="Distance threshold for clustering.")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum number of faces required to form a cluster.")
    
    # Frame Selection (Step 4)
    parser.add_argument("--frame_selection_top_n", type=int, default=5, help="Number of best frames to select per cluster.")
    parser.add_argument("--save_selected_frames", action="store_true", help="Save selected frames as images.")
    
    # Output Generation (Step 5)
    parser.add_argument("--save_scene_clips", action="store_true", help="Save video clips for each scene.")
    parser.add_argument("--draw_selected_frames_on_final_video", action="store_true", help="Draw markers for selected frames on the final video.")

    args = parser.parse_args()
    main(args)