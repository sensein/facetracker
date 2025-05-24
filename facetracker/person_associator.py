import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

from facetracker.mmpose_estimator import MMPoseEstimator

class PersonAssociator:
    """
    Associates detected faces with detected full-body poses on a per-frame basis,
    with temporal consistency and improved occlusion handling.
    """
    def __init__(self, 
                 pose_estimator: MMPoseEstimator, 
                 face_to_pose_iou_threshold: float = 0.3,
                 temporal_window: int = 5,
                 min_pose_confidence: float = 0.3):
        """
        Initializes the PersonAssociator.

        Args:
            pose_estimator (MMPoseEstimator): An initialized instance of the MMPoseEstimator class.
            face_to_pose_iou_threshold (float): Minimum IoU to consider a face and pose matched.
            temporal_window (int): Number of frames to look back for temporal consistency.
            min_pose_confidence (float): Minimum confidence threshold for pose keypoints.
        """
        self.pose_estimator = pose_estimator
        self.face_to_pose_iou_threshold = face_to_pose_iou_threshold
        self.temporal_window = temporal_window
        self.min_pose_confidence = min_pose_confidence
        
        if not isinstance(self.pose_estimator, MMPoseEstimator):
            raise ValueError("pose_estimator must be an instance of the MMPoseEstimator class.")
            
        # Initialize temporal tracking
        self.pose_tracks = defaultdict(list)  # track_id -> list of (frame_idx, pose_data)
        self.next_track_id = 0
        
        print(f"PersonAssociator initialized with:")
        print(f"- IoU threshold: {self.face_to_pose_iou_threshold}")
        print(f"- Temporal window: {self.temporal_window}")
        print(f"- Min pose confidence: {self.min_pose_confidence}")

    @staticmethod
    def _calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / (float(boxAArea + boxBArea - interArea) + 1e-6)
        return iou

    def _calculate_pose_similarity(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        Calculate similarity between two poses using keypoint positions.
        
        Args:
            pose1: First pose keypoints (N, 2)
            pose2: Second pose keypoints (N, 2)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if pose1.shape != pose2.shape:
            return 0.0
            
        # Calculate normalized distances between corresponding keypoints
        distances = np.linalg.norm(pose1 - pose2, axis=1)
        max_distance = np.sqrt(pose1.shape[0])  # Normalize by sqrt of number of keypoints
        similarity = 1.0 - np.mean(distances) / max_distance
        return max(0.0, min(1.0, similarity))

    def _update_pose_tracks(self, frame_idx: int, poses: List[Dict[str, Any]]) -> None:
        """
        Update pose tracks with new detections using temporal consistency.
        
        Args:
            frame_idx: Current frame index
            poses: List of detected poses with keypoints and scores
        """
        # Remove old tracks
        current_tracks = {}
        for track_id, track in self.pose_tracks.items():
            if frame_idx - track[-1][0] <= self.temporal_window:
                current_tracks[track_id] = track
        self.pose_tracks = current_tracks

        # Match new poses to existing tracks
        matched_poses = set()
        for track_id, track in self.pose_tracks.items():
            last_pose = track[-1][1]
            best_match = None
            best_similarity = 0.0
            
            for i, pose in enumerate(poses):
                if i in matched_poses:
                    continue
                    
                similarity = self._calculate_pose_similarity(
                    last_pose['keypoints'],
                    pose['keypoints']
                )
                
                if similarity > best_similarity and similarity > 0.5:
                    best_similarity = similarity
                    best_match = i
            
            if best_match is not None:
                self.pose_tracks[track_id].append((frame_idx, poses[best_match]))
                matched_poses.add(best_match)

        # Create new tracks for unmatched poses
        for i, pose in enumerate(poses):
            if i not in matched_poses:
                self.pose_tracks[self.next_track_id] = [(frame_idx, pose)]
                self.next_track_id += 1

    def process_and_associate(
        self, 
        all_faces_data_by_frame_str: Dict[str, List[Dict[str, Any]]], 
        video_path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processes frames that have face detections, performs pose estimation,
        and associates faces with poses using temporal consistency.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path} in PersonAssociator.")
            return all_faces_data_by_frame_str

        processed_frames_count = 0
        frame_buffer = []  # Store recent frames for temporal consistency
        
        for frame_key_str, faces_in_frame_list in tqdm(all_faces_data_by_frame_str.items(), desc="Associating Faces with Poses"):
            if not faces_in_frame_list:
                continue

            try:
                frame_idx = int(frame_key_str.split("_")[-1])
            except ValueError:
                print(f"Warning: Could not parse frame index from key '{frame_key_str}'. Skipping.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}. Skipping pose association.")
                for face_entry in faces_in_frame_list:
                    face_entry.update({
                        "mmpose_keypoints": None,
                        "mmpose_keypoint_scores": None,
                        "matched_mmpose_head_bbox": None,
                        "face_mmpose_iou": 0.0,
                        "pose_track_id": None,
                        "pose_confidence": 0.0
                    })
                continue
            
            processed_frames_count += 1
            frame_height, frame_width = frame_bgr.shape[:2]
            
            # Extract face bboxes for this frame to provide to MMPose
            face_bboxes = []
            for face_entry in faces_in_frame_list:
                face_bbox = face_entry["bbox"]  # [x1, y1, x2, y2]
                face_bboxes.append(face_bbox)
            
            # Perform pose estimation with face bboxes to avoid problematic detector
            try:
                if face_bboxes:
                    # Provide face bboxes to MMPose to skip internal person detection
                    mmpose_results = self.pose_estimator.estimate_poses(frame_bgr, bboxes=face_bboxes)
                else:
                    # Fallback to full frame detection if no faces
                    mmpose_results = self.pose_estimator.estimate_poses(frame_bgr)
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"CUDA error in pose estimation for frame {frame_idx}, skipping pose estimation for this frame: {e}")
                    mmpose_results = []
                else:
                    raise e
            
            # Process poses and update tracks
            current_poses = []
            for pose_data in mmpose_results:
                if not hasattr(pose_data, 'pred_instances'):
                    continue
                    
                keypoints = pose_data.pred_instances.keypoints
                scores = pose_data.pred_instances.keypoint_scores
                
                if keypoints is None or scores is None:
                    continue
                    
                # Filter poses by confidence
                valid_poses = []
                for kpts, kpt_scores in zip(keypoints, scores):
                    if np.mean(kpt_scores) > self.min_pose_confidence:
                        valid_poses.append({
                            'keypoints': kpts,
                            'scores': kpt_scores,
                            'head_bbox': MMPoseEstimator.get_pose_head_bbox(
                                kpts, kpt_scores,
                                (frame_height, frame_width),
                                keypoint_convention=self.pose_estimator.keypoint_convention
                            )
                        })
                current_poses.extend(valid_poses)
            
            # Update pose tracks with temporal consistency
            self._update_pose_tracks(frame_idx, current_poses)
            
            # Associate faces with poses
            for face_entry in faces_in_frame_list:
                face_bbox = face_entry["bbox"]
                best_match = None
                best_iou = 0.0
                best_track_id = None
                
                # Try to match with current poses
                for pose in current_poses:
                    if pose['head_bbox'] is None:
                        continue
                        
                    iou = self._calculate_iou(face_bbox, pose['head_bbox'])
                    if iou > best_iou and iou >= self.face_to_pose_iou_threshold:
                        best_iou = iou
                        best_match = pose
                
                # If no good match found, try to match with tracked poses
                if best_match is None:
                    for track_id, track in self.pose_tracks.items():
                        if not track:  # Skip empty tracks
                            continue
                            
                        # Get the most recent pose in the track
                        last_pose = track[-1][1]
                        if last_pose['head_bbox'] is None:
                            continue
                            
                        iou = self._calculate_iou(face_bbox, last_pose['head_bbox'])
                        if iou > best_iou and iou >= self.face_to_pose_iou_threshold:
                            best_iou = iou
                            best_match = last_pose
                            best_track_id = track_id
                
                # Update face entry with pose information
                if best_match is not None:
                    face_entry.update({
                        "mmpose_keypoints": best_match['keypoints'].tolist(),
                        "mmpose_keypoint_scores": best_match['scores'].tolist(),
                        "matched_mmpose_head_bbox": best_match['head_bbox'],
                        "face_mmpose_iou": best_iou,
                        "pose_track_id": best_track_id,
                        "pose_confidence": float(np.mean(best_match['scores']))
                    })
                else:
                    face_entry.update({
                        "mmpose_keypoints": None,
                        "mmpose_keypoint_scores": None,
                        "matched_mmpose_head_bbox": None,
                        "face_mmpose_iou": 0.0,
                        "pose_track_id": None,
                        "pose_confidence": 0.0
                    })
        
        cap.release()
        print(f"PersonAssociator processed {processed_frames_count} frames for pose association.")
        return all_faces_data_by_frame_str 