import torch # Ensure torch is imported
# from torch.serialization import add_safe_globals # Direct import
import numpy
import numpy.core.multiarray

# More comprehensive workaround for PyTorch 2.6+ torch.load issue
_globals_to_add_to_torch = []

# Add numpy's _reconstruct
if hasattr(numpy.core.multiarray, '_reconstruct'):
    _globals_to_add_to_torch.append(numpy.core.multiarray._reconstruct)
elif hasattr(numpy.core._multiarray_umath, '_reconstruct'): # Fallback path
    _globals_to_add_to_torch.append(numpy.core._multiarray_umath._reconstruct)

# Add ndarray and top-level dtype
_globals_to_add_to_torch.extend([numpy.ndarray, numpy.dtype])

# Add numpy.core.multiarray.scalar if it exists
if hasattr(numpy.core.multiarray, 'scalar'):
    _globals_to_add_to_torch.append(numpy.core.multiarray.scalar)

# Add all general numpy dtypes (like np.float32, np.int64 etc.)
for name in dir(numpy):
    attr = getattr(numpy, name)
    if isinstance(attr, type) and issubclass(attr, numpy.generic):
        if attr not in _globals_to_add_to_torch:
            _globals_to_add_to_torch.append(attr)

# Add specific numpy.dtypes from numpy.dtypes.__all__ (like UInt8DType)
if hasattr(numpy, 'dtypes') and hasattr(numpy.dtypes, '__all__'):
    for dtype_name in numpy.dtypes.__all__:
        if hasattr(numpy.dtypes, dtype_name):
            attr = getattr(numpy.dtypes, dtype_name)
            # Check if it's a type and not already added
            if isinstance(attr, type) and attr not in _globals_to_add_to_torch:
                _globals_to_add_to_torch.append(attr)
        elif hasattr(numpy, dtype_name): # Fallback to check top-level numpy
            attr = getattr(numpy, dtype_name)
            if isinstance(attr, type) and attr not in _globals_to_add_to_torch:
                 _globals_to_add_to_torch.append(attr)

# if _globals_to_add_to_torch:
#     add_safe_globals(_globals_to_add_to_torch)
#     print(f"DEBUG: Added the following globals to torch safe list: {_globals_to_add_to_torch}")

import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil # For copying files
import json # For saving cluster data
import argparse # Added for command-line arguments

# Import necessary components from the facetracker package
from facetracker.face_detector import FaceDetector
# from facetracker.pose_estimator import PoseEstimator # Old MediaPipe Pose Estimator
from facetracker.mmpose_estimator import MMPoseEstimator # New MMPose Estimator
from facetracker.person_associator import PersonAssociator
from facetracker.face_tracker import FaceTracker, FrameSelector
from facetracker.face_cluster import FaceEmbedder, FaceClusterer

# Configuration (some will become args)
# TEST_VIDEO_PATH = "tests/friends_s02e09a_1min_slice.mp4" # Will be an arg
BASE_OUTPUT_DIR = "tests/output/clustering_visual_test/"
CROPPED_FACES_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "selected_cro_faces") # Where FrameSelector saves its crops
CLUSTERED_FACES_VISUAL_DIR = os.path.join(BASE_OUTPUT_DIR, "clustered_faces_by_id")
CLUSTER_INFO_FILE = os.path.join(BASE_OUTPUT_DIR, "cluster_information.json")

# Model paths (some will become args)
# FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task" # Will be an arg
# POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_lite.task" # Old, will be replaced by MMPose args
# FaceEmbedder uses a default InceptionResnetV1 model from facenet-pytorch (downloads automatically)

# Pipeline Parameters (can also be args if needed)
FRAME_SELECTOR_TOP_N = 5 # Select top 5 frames per track for embedding/clustering
CLUSTERER_SIMILARITY_THRESHOLD = 0.5 # Adjust as needed

def ensure_output_dirs():
    os.makedirs(CROPPED_FACES_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(CLUSTERED_FACES_VISUAL_DIR, exist_ok=True) # This line is redundant due to rmtree then makedirs
    # Clean previous clustering results for a fresh run
    if os.path.exists(CLUSTERED_FACES_VISUAL_DIR):
        shutil.rmtree(CLUSTERED_FACES_VISUAL_DIR)
    os.makedirs(CLUSTERED_FACES_VISUAL_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Test script for visual face clustering with MMPose.")
    parser.add_argument(
        "--video_path",
        type=str,
        default="tests/friends_s02e09a_1min_slice.mp4",
        help="Path to the test video file."
    )
    parser.add_argument(
        "--output_dir_base", # Renamed from BASE_OUTPUT_DIR for clarity
        type=str,
        default="tests/output/clustering_visual_test/",
        help="Base directory to save output files."
    )
    # FaceDetector args
    parser.add_argument(
        "--face_landmarker_model_path", 
        type=str, 
        default="face_landmarker.task",
        help="Path to the MediaPipe FaceLandmarker model (.task file)."
    )
    parser.add_argument(
        "--deepface_backend",
        type=str,
        default='retinaface',
        choices=['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface', 'humaneight'],
        help="Backend for DeepFace face detector."
    )
    # MMPoseEstimator args
    parser.add_argument(
        "--mmpose_model_config",
        type=str, 
        default="mmpose_models/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        help="Path to MMPose pose model config file (for MMPoseEstimator)."
    )
    parser.add_argument(
        "--mmpose_model_checkpoint",
        type=str, 
        default="mmpose_models/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth",
        help="Path to MMPose pose model checkpoint file (for MMPoseEstimator)."
    )
    parser.add_argument(
        "--mmpose_keypoint_convention",
        type=str,
        default="coco",
        help="Keypoint convention for the loaded MMPose model (e.g., 'coco')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run pose estimation on (e.g., 'cuda:0' or 'cpu')."
    )
    # Pipeline parameters as args
    parser.add_argument(
        "--frame_selector_top_n",
        type=int,
        default=FRAME_SELECTOR_TOP_N,
        help="Number of top frames to select per track for embedding."
    )
    parser.add_argument(
        "--clusterer_similarity_threshold",
        type=float,
        default=CLUSTERER_SIMILARITY_THRESHOLD,
        help="Similarity threshold for face clustering."
    )

    args = parser.parse_args()

    # Update output paths based on args.output_dir_base
    global BASE_OUTPUT_DIR, CROPPED_FACES_OUTPUT_DIR, CLUSTERED_FACES_VISUAL_DIR, CLUSTER_INFO_FILE
    BASE_OUTPUT_DIR = args.output_dir_base
    CROPPED_FACES_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "selected_cropped_faces")
    CLUSTERED_FACES_VISUAL_DIR = os.path.join(BASE_OUTPUT_DIR, "clustered_faces_by_id")
    CLUSTER_INFO_FILE = os.path.join(BASE_OUTPUT_DIR, "cluster_information.json")
    
    ensure_output_dirs()

    # --- 1. Initialization --- 
    print("Initializing pipeline components...")
    # Check all model paths from args
    required_model_paths = [
        args.face_landmarker_model_path,
        args.mmpose_model_config, args.mmpose_model_checkpoint
    ]
    if not all(os.path.exists(p) for p in required_model_paths):
        print(f"ERROR: One or more model files not found. Checked paths:")
        for p in required_model_paths:
            if not os.path.exists(p): print(f" - {p} (MISSING)")
        return

    face_detector = FaceDetector(
        video_path=args.video_path, 
        output_dir=os.path.join(BASE_OUTPUT_DIR, "face_detector_temp"), # Temporary, can be improved
        face_landmarker_model_path=args.face_landmarker_model_path,
        deepface_backend=args.deepface_backend
    )
    
    print(f"Initializing MMPoseEstimator with device: {args.device}")
    pose_estimator = MMPoseEstimator(
        pose_model_config=args.mmpose_model_config,
        pose_model_checkpoint=args.mmpose_model_checkpoint,
        device=args.device,
        keypoint_convention=args.mmpose_keypoint_convention
    )
    person_associator = PersonAssociator(pose_estimator=pose_estimator) # Use the MMPoseEstimator
    face_tracker = FaceTracker() # Default SORT params
    
    frame_selector = FrameSelector(
        video_file=args.video_path, 
        top_n=args.frame_selector_top_n, 
        output_dir=CROPPED_FACES_OUTPUT_DIR, 
        save_images=True
    )
    face_embedder = FaceEmbedder() # Default model
    face_clusterer = FaceClusterer(similarity_threshold=args.clusterer_similarity_threshold)

    # --- 2. Video Processing (Detection, Association, Tracking) --- 
    print(f"Processing video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened(): print(f"Error opening video {args.video_path}"); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Not used
    cap.release() 

    print("Step 1: Running Face Detection...")
    all_faces_by_frame_str = face_detector.detect_faces_in_video()
    if not all_faces_by_frame_str: print("No faces detected. Exiting."); return

    print("Step 2: Running Person Association (Face+MMPose)...")
    augmented_data_by_frame_str = person_associator.process_and_associate(
        all_faces_by_frame_str, args.video_path
    )

    print("Step 3: Running Face Tracking...")
    tracked_instances_list = face_tracker.track_faces(
        augmented_data_by_frame_str, frame_width, video_height=frame_height
    )
    if not tracked_instances_list: print("No faces tracked. Exiting."); return

    scene_id = "scene_0" # Assuming single scene for this test
    tracked_data_for_selector = {scene_id: tracked_instances_list}

    # --- 3. Frame Selection --- 
    print("Step 4: Running Frame Selection...")
    selected_frames_by_scene = frame_selector.select_top_frames_per_face(tracked_data_for_selector)
    if not selected_frames_by_scene.get(scene_id):
        print("No frames selected by FrameSelector. Exiting."); return

    # --- 4. Face Embedding --- 
    print("Step 5: Running Face Embedding...")
    all_tracks_data_with_embeddings = face_embedder.get_face_embeddings(
        selected_frames_by_scene, CROPPED_FACES_OUTPUT_DIR # Pass absolute path
    )
    if not all_tracks_data_with_embeddings: print("No embeddings generated. Exiting."); return

    # --- 5. Face Clustering --- 
    print("Step 6: Running Face Clustering...")
    final_clusters = face_clusterer.cluster_faces(all_tracks_data_with_embeddings)
    if not final_clusters: print("No clusters generated. Exiting."); return

    # --- 6. Visualizing Clusters by Copying Images --- 
    print(f"Step 7: Visualizing clusters in {CLUSTERED_FACES_VISUAL_DIR}...")
    cluster_summary_for_json = {}

    for cluster_label, nodes_in_cluster in final_clusters.items():
        cluster_dir = os.path.join(CLUSTERED_FACES_VISUAL_DIR, f"cluster_{cluster_label}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        cluster_summary_for_json[str(cluster_label)] = {
            "count": len(nodes_in_cluster),
            "image_paths": []
        }

        for node_data in nodes_in_cluster:
            # image_path from FaceEmbedder is relative to its input image_dir (CROPPED_FACES_OUTPUT_DIR)
            relative_image_path = node_data.get("image_path") 
            if relative_image_path:
                src_image_full_path = os.path.join(CROPPED_FACES_OUTPUT_DIR, relative_image_path)
                if os.path.exists(src_image_full_path):
                    dest_filename = os.path.basename(relative_image_path) 
                    dest_image_full_path = os.path.join(cluster_dir, dest_filename)
                    shutil.copy(src_image_full_path, dest_image_full_path)
                    cluster_summary_for_json[str(cluster_label)]["image_paths"].append(relative_image_path)
                else:
                    print(f"Warning: Source image for cluster not found: {src_image_full_path}")
            else:
                print(f"Warning: Node data in cluster {cluster_label} missing 'image_path'. Node: {node_data.get('node_id')}")
    
    with open(CLUSTER_INFO_FILE, 'w') as f:
        json.dump(cluster_summary_for_json, f, indent=4)
    print(f"Cluster information saved to {CLUSTER_INFO_FILE}")

    # --- 7. Cleanup (Optional: close any components that need it) --- 
    print("Cleaning up component resources...")
    if hasattr(face_detector, 'close'): face_detector.close()
    # MMPoseEstimator does not have an explicit close method in current design
    # if hasattr(pose_estimator, 'close'): pose_estimator.close() 
    if hasattr(frame_selector, 'close'): frame_selector.close()

    print(f"Finished. Visual clustering results are in {CLUSTERED_FACES_VISUAL_DIR}")
    print("Please manually inspect the subdirectories to verify clustering quality.")

if __name__ == "__main__":
    main() 