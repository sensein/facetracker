import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import mediapipe as mp

# Import the specific protobuf definition needed for drawing utils
from mediapipe.framework.formats import landmark_pb2

# MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# --- Helper Function for Colors ---
def get_cluster_color(cluster_id):
    """Gets a distinct color for a given cluster ID."""
    cmap = plt.get_cmap('tab20') # Use a colormap with more distinct colors
    return tuple(int(c * 255) for c in cmap(cluster_id % 20)[:3]) # Modulo 20 for cmap size

# --- Helper to convert list data back to LandmarkList proto ---
def to_landmark_list_proto(landmarks_data: list, is_pose=False) -> landmark_pb2.NormalizedLandmarkList:
    """Converts list of [x,y,z] or [x,y,z,v] back to NormalizedLandmarkList proto."""
    landmark_list_proto = landmark_pb2.NormalizedLandmarkList()
    if landmarks_data:
        for lm_data in landmarks_data:
            # Check visibility if available (pose landmarks have it)
            visibility = lm_data[3] if is_pose and len(lm_data) > 3 else None
            landmark_list_proto.landmark.add(
                x=lm_data[0], y=lm_data[1], z=lm_data[2], visibility=visibility
            )
    return landmark_list_proto

# --- Helper class for JSON deserialization ---
class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # This is a basic hook; might need refinement if embeddings aren't just top-level lists
        for key, value in obj.items():
            if isinstance(value, list) and key == "embedding": # Simple check for embedding key
                 try:
                     # Attempt conversion if it looks like numerical data
                     arr = np.array(value)
                     if np.issubdtype(arr.dtype, np.number):
                         obj[key] = arr
                 except ValueError:
                     pass # Keep as list if conversion fails
        return obj

# --- Main Visualization Function ---
def visualize_final_clusters(video_path: str, cluster_json_path: str, output_video_path: str,
                             draw_face_mesh: bool = True, draw_pose: bool = True):
    """
    Loads final clustering results and creates an annotated video.
    """
    print(f"Starting cluster visualization for video: {video_path}")
    print(f"Using cluster data from: {cluster_json_path}")
    print(f"Output video will be saved to: {output_video_path}")

    # --- 1. Load Cluster Data ---
    print("Step 1: Loading cluster data...")
    try:
        with open(cluster_json_path, 'r') as f:
            # Using standard json load for now, assuming embeddings aren't needed for viz
            # If you need embeddings later, use json.load(f, cls=NumpyDecoder)
             final_clusters = json.load(f) # Keys are stringified cluster IDs
    except FileNotFoundError:
        print(f"Error: Cluster JSON file not found at {cluster_json_path}")
        return
    except json.JSONDecodeError as e:
         print(f"Error: Failed to decode JSON file {cluster_json_path}: {e}")
         return
    print(f"Loaded data for {len(final_clusters)} clusters.")

    # --- 2. Preprocess Cluster Data for Frame Lookup ---
    print("Step 2: Preprocessing cluster data for frame lookup...")
    frame_to_persons = {} # Key: frame_idx, Value: List[person_instance_data]
    for cluster_id_str, instances in final_clusters.items():
        cluster_id_int = int(cluster_id_str) # Cluster ID is the global person ID
        for instance_data in instances:
            frame_idx = instance_data["frame_idx"]
            if frame_idx not in frame_to_persons:
                frame_to_persons[frame_idx] = []
            # Add the global cluster ID to the instance data for easy access
            instance_data['global_person_id'] = cluster_id_int
            frame_to_persons[frame_idx].append(instance_data)
    print("Preprocessing complete.")

    # --- 3. Setup Video Reading/Writing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path} for reading.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height))
    if not out_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return

    # --- 4. Process Video Frames and Annotate ---
    print("Step 3: Processing video frames and annotating clusters...")
    for frame_idx in tqdm(range(total_frames), desc="Visualizing Clusters", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}. Stopping.")
            break

        persons_in_frame = frame_to_persons.get(frame_idx, [])

        for person_data in persons_in_frame:
            cluster_id = person_data['global_person_id']
            color = get_cluster_color(cluster_id)
            bbox = person_data.get('face_coord')
            face_mesh_data = person_data.get('face_mesh')
            pose_data = person_data.get('full_body_pose')

            # Draw BBox and ID Label
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Person: {cluster_id}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y1_label = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10),
                              (x1 + label_size[0], y1_label - base_line), color, cv2.FILLED)
                cv2.putText(frame, label, (x1, y1_label - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text

            # Draw Face Mesh
            if draw_face_mesh and face_mesh_data:
                 face_landmarks_proto = to_landmark_list_proto(face_mesh_data, is_pose=False)
                 mp_drawing.draw_landmarks(
                     image=frame,
                     landmark_list=face_landmarks_proto,
                     connections=mp_face_mesh.FACEMESH_TESSELATION,
                     landmark_drawing_spec=None, # Default small dots
                     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                     # For colored connections: mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1))
                     # Styles might need adjustment for custom colors

            # Draw Body Pose
            if draw_pose and pose_data:
                 pose_landmarks_proto = to_landmark_list_proto(pose_data, is_pose=True)
                 mp_drawing.draw_landmarks(
                     image=frame,
                     landmark_list=pose_landmarks_proto,
                     connections=mp_pose.POSE_CONNECTIONS,
                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                     # For colored landmarks: mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))

        # Write annotated frame
        out_writer.write(frame)

    # --- 5. Release Resources ---
    print("Releasing resources...")
    cap.release()
    out_writer.release()
    print(f"Cluster visualization saved to {output_video_path}")


# --- Argument Parser ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize final face clustering results.")
    parser.add_argument("--video_path", default="/orcd/data/satra/002/datasets/cneuromod/friends/stimuli/s1/friends_s01e01a.mkv", help="Path to the original input video file.")
    parser.add_argument("--cluster_json", default="facetracker_pipeline_output/final_face_clusters.json", help="Path to the final_face_clusters.json file.")
    parser.add_argument("--output_video", default="facetracker_pipeline_output/video_tracking_clusters.mp4", help="Path to save the output annotated video.")
    parser.add_argument("--no-face-mesh", action="store_true", help="Disable drawing of face mesh landmarks.")
    parser.add_argument("--no-pose", action="store_true", help="Disable drawing of pose landmarks.")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Input video not found at {args.video_path}")
    elif not os.path.exists(args.cluster_json):
         print(f"Error: Cluster JSON file not found at {args.cluster_json}")
    else:
        visualize_final_clusters(args.video_path, args.cluster_json, args.output_video,
                                 draw_face_mesh=not args.no_face_mesh,
                                 draw_pose=not args.no_pose)