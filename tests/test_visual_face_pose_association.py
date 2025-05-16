import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import argparse

from facetracker.face_detector import FaceDetector
from facetracker.mmpose_estimator import MMPoseEstimator
from facetracker.person_associator import PersonAssociator
from facetracker.face_tracker import FaceTracker

# Configuration
OUTPUT_VIDEO_FILENAME = "annotated_face_pose_association.mp4"

# Drawing Configuration
FACE_BBOX_COLOR = (0, 255, 0)  # Green
FACE_ID_COLOR = (0, 255, 0)
POSE_LANDMARK_COLOR = (255, 0, 0) # Blue
POSE_HEAD_BBOX_COLOR = (0, 0, 255) # Red
ASSOCIATION_LINE_COLOR = (255, 255, 0) # Cyan
TEXT_SCALE = 0.7
TEXT_THICKNESS = 1

def ensure_output_dir(output_dir_path: str):
    os.makedirs(output_dir_path, exist_ok=True)

def draw_results_on_frame(frame_bgr: np.ndarray, 
                         tracked_faces_in_frame: List[Dict[str, Any]],
                         all_poses_in_frame: Optional[List[Dict[str, Any]]],
                         associator: PersonAssociator,
                         frame_width: int, frame_height: int) -> np.ndarray:
    """Draws face detections, track IDs, poses, and associations on the frame."""
    annotated_frame = frame_bgr.copy()

    # Draw poses first, so they are underneath face boxes if they overlap
    if all_poses_in_frame:
        for pose_data in all_poses_in_frame:
            keypoints = pose_data.get('keypoints')
            scores = pose_data.get('scores')
            if keypoints is None or scores is None:
                continue

            # Draw keypoints
            for kpt, score in zip(keypoints, scores):
                if score > associator.min_pose_confidence:
                    x = int(kpt[0] * frame_width)
                    y = int(kpt[1] * frame_height)
                    cv2.circle(annotated_frame, (x, y), 2, POSE_LANDMARK_COLOR, -1)
            
            # Draw head bbox
            head_bbox = pose_data.get('head_bbox')
            if head_bbox:
                cv2.rectangle(annotated_frame, 
                            (int(head_bbox[0]), int(head_bbox[1])), 
                            (int(head_bbox[2]), int(head_bbox[3])), 
                            POSE_HEAD_BBOX_COLOR, 1)

    # Draw faces and their associations
    for face_data in tracked_faces_in_frame:
        face_bbox = face_data.get("bbox")
        track_id = face_data.get("id", -1)
        confidence = face_data.get("confidence", 0.0)

        if face_bbox:
            x1, y1, x2, y2 = map(int, face_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), FACE_BBOX_COLOR, 2)
            label = f"ID: {track_id} C: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, FACE_ID_COLOR, TEXT_THICKNESS)

            # Check if this face was associated with a pose
            matched_pose_head_bbox = face_data.get("matched_mmpose_head_bbox")
            face_pose_iou = face_data.get("face_mmpose_iou", 0.0)
            
            if matched_pose_head_bbox and face_pose_iou >= associator.face_to_pose_iou_threshold:
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2
                
                ph_x1, ph_y1, ph_x2, ph_y2 = map(int, matched_pose_head_bbox)
                pose_head_center_x = (ph_x1 + ph_x2) // 2
                pose_head_center_y = (ph_y1 + ph_y2) // 2
                
                cv2.line(annotated_frame, (face_center_x, face_center_y), 
                         (pose_head_center_x, pose_head_center_y), 
                         ASSOCIATION_LINE_COLOR, 1)
                iou_text = f"IoU: {face_pose_iou:.2f}"
                cv2.putText(annotated_frame, iou_text, (x1, y2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ASSOCIATION_LINE_COLOR, TEXT_THICKNESS)

    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="Test script for face and pose association visualization.")
    parser.add_argument(
        "--face_detection_model_path", 
        type=str, 
        default=None,
        help="Path to the face detection model. If None, uses default."
    )
    parser.add_argument(
        "--mmpose_model_type", 
        type=str, 
        default="hiera_l",
        help="Type of MMPose model to use."
    )
    parser.add_argument(
        "--mmpose_config_path", 
        type=str, 
        required=True,
        help="Path to the MMPose model configuration file."
    )
    parser.add_argument(
        "--mmpose_checkpoint_path", 
        type=str, 
        required=True,
        help="Path to the MMPose model checkpoint file."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="tests/friends_s02e09a_1min_slice.mp4",
        help="Path to the test video file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/output/",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for models ('cuda' or 'cpu')."
    )
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)
    output_video_path = os.path.join(args.output_dir, OUTPUT_VIDEO_FILENAME)

    # 1. Initialize Components
    print("Initializing components...")
    face_detector = FaceDetector(
        model_path=args.face_detection_model_path,
        device=args.device,
        output_dir=os.path.join(args.output_dir, "face_detector_temp")
    )
    
    mmpose_estimator = MMPoseEstimator(
        model_type=args.mmpose_model_type,
        model_config_path=args.mmpose_config_path,
        checkpoint_path=args.mmpose_checkpoint_path,
        device=args.device
    )
    
    person_associator = PersonAssociator(
        pose_estimator=mmpose_estimator,
        face_to_pose_iou_threshold=0.3,
        temporal_window=5,
        min_pose_confidence=0.3
    )
    
    face_tracker = FaceTracker(
        iou_threshold=0.4,
        max_lost_tracks=10,
        min_track_length=5,
        use_pose_info=True
    )

    # 2. Open Video and VideoWriter
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {args.video_path}")
    print(f"Output will be saved to: {output_video_path}")

    # 3. Process Video Frame by Frame
    all_augmented_faces_by_frame_str_for_tracking: Dict[str, List[Dict[str, Any]]] = {}
    all_pose_results_by_frame_idx: Dict[int, List[Dict[str, Any]]] = {}

    print("Pass 1: Detecting faces, poses, and associating...")
    for i in tqdm(range(total_frames), desc="Detecting & Associating"):
        ret, frame_bgr = cap.read()
        if not ret: break

        # Detect faces
        detected_faces_list_current_frame = face_detector._detect_faces_and_landmarks_frame(frame_bgr)
        
        # Detect poses
        pose_results = mmpose_estimator.estimate_poses(frame_bgr)
        all_pose_results_by_frame_idx[i] = pose_results
        
        # Associate faces with poses
        current_frame_face_data_for_assoc = {f"frame_{i}": detected_faces_list_current_frame}
        augmented_faces_this_frame = []
        
        if detected_faces_list_current_frame:
            for face_entry in detected_faces_list_current_frame:
                current_face_augmented = face_entry.copy()
                current_face_augmented.update({
                    "mmpose_keypoints": None,
                    "mmpose_keypoint_scores": None,
                    "matched_mmpose_head_bbox": None,
                    "face_mmpose_iou": 0.0,
                    "pose_track_id": None,
                    "pose_confidence": 0.0
                })
                
                best_iou_for_this_face = 0.0
                if pose_results:
                    for pose_data in pose_results:
                        if not hasattr(pose_data, 'pred_instances'):
                            continue
                            
                        keypoints = pose_data.pred_instances.keypoints
                        scores = pose_data.pred_instances.keypoint_scores
                        
                        if keypoints is None or scores is None:
                            continue
                            
                        head_bbox = mmpose_estimator.get_pose_head_bbox(
                            keypoints[0], scores[0],
                            (frame_height, frame_width)
                        )
                        
                        if head_bbox is None:
                            continue
                            
                        iou = person_associator._calculate_iou(face_entry["bbox"], head_bbox)
                        if iou > best_iou_for_this_face:
                            best_iou_for_this_face = iou
                            if iou >= person_associator.face_to_pose_iou_threshold:
                                current_face_augmented.update({
                                    "mmpose_keypoints": keypoints[0].tolist(),
                                    "mmpose_keypoint_scores": scores[0].tolist(),
                                    "matched_mmpose_head_bbox": head_bbox,
                                    "face_mmpose_iou": iou,
                                    "pose_confidence": float(np.mean(scores[0]))
                                })
                
                current_face_augmented["face_mmpose_iou"] = best_iou_for_this_face
                augmented_faces_this_frame.append(current_face_augmented)
        
        all_augmented_faces_by_frame_str_for_tracking[f"frame_{i}"] = augmented_faces_this_frame

    # Track faces
    print("\nPass 2: Tracking faces...")
    tracked_instances_all_frames_list = face_tracker.track_faces(
        all_augmented_faces_by_frame_str_for_tracking,
        frame_width,
        frame_height
    )

    # Group tracked instances by frame
    tracked_data_grouped_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for tracked_instance in tracked_instances_all_frames_list:
        f_idx = tracked_instance["frame"]
        if f_idx not in tracked_data_grouped_by_frame:
            tracked_data_grouped_by_frame[f_idx] = []
        tracked_data_grouped_by_frame[f_idx].append(tracked_instance)

    # Draw results
    print("\nDrawing Pass: Writing output video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(total_frames), desc="Drawing Video"):
        ret, frame_bgr = cap.read()
        if not ret: break

        faces_to_draw_this_frame = tracked_data_grouped_by_frame.get(i, [])
        poses_to_draw_this_frame = all_pose_results_by_frame_idx.get(i, [])
        
        annotated_frame = draw_results_on_frame(
            frame_bgr, 
            faces_to_draw_this_frame, 
            poses_to_draw_this_frame, 
            person_associator,
            frame_width, 
            frame_height
        )
        video_writer.write(annotated_frame)

    # 4. Cleanup
    print("Cleaning up...")
    cap.release()
    video_writer.release()
    mmpose_estimator.close()
    print(f"Finished. Annotated video saved to {output_video_path}")

if __name__ == "__main__":
    main() 