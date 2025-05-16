import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil # For copying files
import json # For saving cluster data

# Import necessary components from the facetracker package
from facetracker.face_detector import FaceDetector
from facetracker.pose_estimator import PoseEstimator
from facetracker.person_associator import PersonAssociator
from facetracker.face_tracker import FaceTracker, FrameSelector
from facetracker.face_cluster import FaceEmbedder, FaceClusterer

# Configuration
TEST_VIDEO_PATH = "tests/friends_s02e09a_1min_slice.mp4"
BASE_OUTPUT_DIR = "tests/output/clustering_visual_test/"
CROPPED_FACES_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "selected_cro_faces") # Where FrameSelector saves its crops
CLUSTERED_FACES_VISUAL_DIR = os.path.join(BASE_OUTPUT_DIR, "clustered_faces_by_id")
CLUSTER_INFO_FILE = os.path.join(BASE_OUTPUT_DIR, "cluster_information.json")

# Model paths (ensure these are accessible)
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"
POSE_LANDMARKER_MODEL_PATH = "pose_landmarker_lite.task"
# FaceEmbedder uses a default InceptionResnetV1 model from facenet-pytorch (downloads automatically)

# Pipeline Parameters
FRAME_SELECTOR_TOP_N = 5 # Select top 5 frames per track for embedding/clustering
CLUSTERER_SIMILARITY_THRESHOLD = 0.5 # Adjust as needed

def ensure_output_dirs():
    os.makedirs(CROPPED_FACES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CLUSTERED_FACES_VISUAL_DIR, exist_ok=True)
    # Clean previous clustering results for a fresh run
    if os.path.exists(CLUSTERED_FACES_VISUAL_DIR):
        shutil.rmtree(CLUSTERED_FACES_VISUAL_DIR)
    os.makedirs(CLUSTERED_FACES_VISUAL_DIR, exist_ok=True)

def main():
    ensure_output_dirs()

    # --- 1. Initialization --- 
    print("Initializing pipeline components...")
    if not all(os.path.exists(p) for p in [FACE_LANDMARKER_MODEL_PATH, POSE_LANDMARKER_MODEL_PATH]):
        print(f"ERROR: One or more model files not found. Checked:",
              f"\n - {FACE_LANDMARKER_MODEL_PATH}",
              f"\n - {POSE_LANDMARKER_MODEL_PATH}")
        return

    face_detector = FaceDetector(
        video_path=TEST_VIDEO_PATH, 
        output_dir=os.path.join(BASE_OUTPUT_DIR, "face_detector_temp"),
        face_landmarker_model_path=FACE_LANDMARKER_MODEL_PATH
    )
    pose_estimator = PoseEstimator(
        model_asset_path=POSE_LANDMARKER_MODEL_PATH,
        running_mode=mp.tasks.python.vision.RunningMode.VIDEO,
        output_segmentation_masks=True 
    )
    person_associator = PersonAssociator(pose_estimator=pose_estimator)
    face_tracker = FaceTracker() # Default SORT params
    
    # FrameSelector needs the output_dir for its cropped images
    frame_selector = FrameSelector(
        video_file=TEST_VIDEO_PATH, 
        top_n=FRAME_SELECTOR_TOP_N, 
        output_dir=CROPPED_FACES_OUTPUT_DIR, 
        save_images=True
    )
    face_embedder = FaceEmbedder() # Default model
    face_clusterer = FaceClusterer(similarity_threshold=CLUSTERER_SIMILARITY_THRESHOLD)

    # --- 2. Video Processing (Detection, Association, Tracking) --- 
    print(f"Processing video: {TEST_VIDEO_PATH}")
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    if not cap.isOpened(): print(f"Error opening video {TEST_VIDEO_PATH}"); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # Release early, individual components will reopen if needed or work on frames directly

    print("Step 1: Running Face Detection...")
    # FaceDetector.detect_faces_in_video() returns Dict[frame_key_str, List[face_detection_dict]]
    all_faces_by_frame_str = face_detector.detect_faces_in_video()
    if not all_faces_by_frame_str: print("No faces detected. Exiting."); return

    print("Step 2: Running Person Association (Face+Pose)...")
    # PersonAssociator.process_and_associate augments the face data in place
    # It expects the video_path to read frames for pose estimation.
    augmented_data_by_frame_str = person_associator.process_and_associate(
        all_faces_by_frame_str, TEST_VIDEO_PATH
    )

    print("Step 3: Running Face Tracking...")
    # FaceTracker.track_faces expects Dict[frame_key_str, List[augmented_face_dict]]
    # and returns List[tracked_instance_dict] where each dict has 'id', 'frame', 'bbox', etc.
    tracked_instances_list = face_tracker.track_faces(
        augmented_data_by_frame_str, frame_width, video_height=frame_height
    )
    if not tracked_instances_list: print("No faces tracked. Exiting."); return

    # Reorganize tracked data by scene (assuming single scene for this test video)
    # The pipeline typically assumes a "scene_0" if not otherwise specified.
    # FrameSelector expects data in format: Dict[scene_id, List[tracked_instance_dict]]
    scene_id = "scene_0"
    tracked_data_for_selector = {scene_id: tracked_instances_list}

    # --- 3. Frame Selection --- 
    print("Step 4: Running Frame Selection...")
    # FrameSelector saves cropped images to its self.output_dir (CROPPED_FACES_OUTPUT_DIR)
    # Output: Dict[scene_id, List[unique_track_data_dict]]
    # unique_track_data_dict: {"unique_track_id", "top_frames": List[frame_info_dict]}
    # frame_info_dict: {"image_path" (relative to CROPPED_FACES_OUTPUT_DIR), ...}
    selected_frames_by_scene = frame_selector.select_top_frames_per_face(tracked_data_for_selector)
    if not selected_frames_by_scene.get(scene_id):
        print("No frames selected by FrameSelector. Exiting."); return

    # --- 4. Face Embedding --- 
    print("Step 5: Running Face Embedding...")
    # FaceEmbedder expects image_dir to be where FrameSelector saved its crops.
    # Output: List[track_data_with_embeddings_dict]
    # track_data_with_embeddings_dict: {"unique_track_id", "scene_id", "frames_data": List[frame_embedding_data_dict]}
    # frame_embedding_data_dict: {"embedding", "image_path", ...}
    all_tracks_data_with_embeddings = face_embedder.get_face_embeddings(
        selected_frames_by_scene, CROPPED_FACES_OUTPUT_DIR
    )
    if not all_tracks_data_with_embeddings: print("No embeddings generated. Exiting."); return

    # --- 5. Face Clustering --- 
    print("Step 6: Running Face Clustering...")
    # Output: Dict[cluster_label (int), List[node_attribute_dict]]
    # node_attribute_dict: {"node_id", "unique_track_id", "scene_id", "embedding", "image_path", ...}
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
            relative_image_path = node_data.get("image_path") # Path relative to CROPPED_FACES_OUTPUT_DIR
            if relative_image_path:
                src_image_full_path = os.path.join(CROPPED_FACES_OUTPUT_DIR, relative_image_path)
                if os.path.exists(src_image_full_path):
                    # Create a unique name for the destination to avoid overwrites if multiple original files had same name (unlikely with FrameSelector naming)
                    # Or, just copy with original name if FrameSelector's naming is guaranteed unique.
                    # For simplicity, using original relative path as part of the name in cluster dir.
                    dest_filename = os.path.basename(relative_image_path) # e.g., scene_X_track_Y_frame_Z.jpg
                    dest_image_full_path = os.path.join(cluster_dir, dest_filename)
                    shutil.copy(src_image_full_path, dest_image_full_path)
                    cluster_summary_for_json[str(cluster_label)]["image_paths"].append(relative_image_path)
                else:
                    print(f"Warning: Source image for cluster not found: {src_image_full_path}")
            else:
                print(f"Warning: Node data in cluster {cluster_label} missing 'image_path'. Node: {node_data.get('node_id')}")
    
    # Save cluster information
    with open(CLUSTER_INFO_FILE, 'w') as f:
        json.dump(cluster_summary_for_json, f, indent=4)
    print(f"Cluster information saved to {CLUSTER_INFO_FILE}")

    # --- 7. Cleanup (Optional: close any components that need it) --- 
    print("Cleaning up component resources...")
    if hasattr(face_detector, 'close'): face_detector.close()
    if hasattr(pose_estimator, 'close'): pose_estimator.close()
    if hasattr(frame_selector, 'close'): frame_selector.close()
    # FaceEmbedder and FaceClusterer don't have explicit close methods in current design.

    print(f"Finished. Visual clustering results are in {CLUSTERED_FACES_VISUAL_DIR}")
    print("Please manually inspect the subdirectories to verify clustering quality.")

if __name__ == "__main__":
    main() 