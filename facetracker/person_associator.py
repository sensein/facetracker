import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import mediapipe as mp

from facetracker.pose_estimator import PoseEstimator

class PersonAssociator:
    """
    Associates detected faces with detected full-body poses on a per-frame basis.
    """
    def __init__(self, pose_estimator: PoseEstimator, face_to_pose_iou_threshold: float = 0.3):
        """
        Initializes the PersonAssociator.

        Args:
            pose_estimator (PoseEstimator): An initialized instance of the PoseEstimator class.
            face_to_pose_iou_threshold (float): Minimum IoU to consider a face and pose matched.
        """
        self.pose_estimator = pose_estimator
        self.face_to_pose_iou_threshold = face_to_pose_iou_threshold
        if not isinstance(self.pose_estimator, PoseEstimator):
            raise ValueError("pose_estimator must be an instance of the PoseEstimator class.")
        print(f"PersonAssociator initialized with IoU threshold: {face_to_pose_iou_threshold}")

    @staticmethod
    def _calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        Boxes are [x1, y1, x2, y2]. Input coordinates can be float or int.
        """
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

    def process_and_associate(
        self, 
        all_faces_data_by_frame_str: Dict[str, List[Dict[str, Any]]], 
        video_path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processes frames that have face detections, performs pose estimation on these specific frames,
        and associates faces with poses.

        Args:
            all_faces_data_by_frame_str: Output from FaceDetector 
                (Dict[frame_key_str, List[face_detection_dict]]).
            video_path: Path to the video file.

        Returns:
            The input all_faces_data_by_frame_str, with face_detection_dict entries
            potentially augmented with 'full_body_pose_landmarks', 
            'matched_pose_head_bbox', 'face_pose_iou', and 'human_segmentation_mask'.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path} in PersonAssociator.")
            return all_faces_data_by_frame_str

        # Create a copy to modify, or modify in place if that's the desired contract
        # For clarity, let's assume we modify in place as the name suggests augmentation.
        # If not, a deep copy would be needed: augmented_data = copy.deepcopy(all_faces_data_by_frame_str)
        
        processed_frames_count = 0
        
        # Iterate through only the frames that have face detections
        for frame_key_str, faces_in_frame_list in tqdm(all_faces_data_by_frame_str.items(), desc="Associating Faces with Poses per Frame"):
            if not faces_in_frame_list: # Skip if no faces were detected in this stored frame_key
                continue

            try:
                # Assuming frame_key_str is like "frame_123"
                frame_idx = int(frame_key_str.split("_")[-1])
            except ValueError:
                print(f"Warning: Could not parse frame index from key '{frame_key_str}'. Skipping this entry.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from video. Skipping pose association for this frame.")
                # Mark existing face entries for this frame as having no pose, or handle as error
                for face_entry in faces_in_frame_list:
                    face_entry["full_body_pose_landmarks"] = None
                    face_entry["matched_pose_head_bbox"] = None
                    face_entry["face_pose_iou"] = 0.0
                    face_entry["human_segmentation_mask"] = None # Initialize new field
                continue
            
            processed_frames_count += 1
            frame_height, frame_width = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Perform pose estimation for the current frame
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) # For VIDEO mode of PoseEstimator
            pose_landmarker_result = None
            if self.pose_estimator.running_mode == mp.tasks.python.vision.RunningMode.VIDEO:
                pose_landmarker_result = self.pose_estimator.detect_pose_in_frame(frame_rgb, timestamp_ms=timestamp_ms)
            else: # Assume IMAGE mode
                pose_landmarker_result = self.pose_estimator.detect_pose_in_frame(frame_rgb)

            # Now, for each face detected in this frame, try to associate with a detected pose
            for face_entry in faces_in_frame_list: # This iterates over list of dicts for the current frame_key_str
                face_bbox_coords = face_entry["bbox"] 
                # Initialize fields for this specific face before trying to match with poses
                face_entry["full_body_pose_landmarks"] = None
                face_entry["matched_pose_head_bbox"] = None
                face_entry["face_pose_iou"] = 0.0
                face_entry["human_segmentation_mask"] = None # Initialize new field for each face

                best_iou_for_this_face = 0.0 # Reset for each face

                if pose_landmarker_result and pose_landmarker_result.pose_landmarks:
                    for pose_idx, single_pose_normalized_landmarks in enumerate(pose_landmarker_result.pose_landmarks):
                        current_pose_head_bbox = PoseEstimator.get_pose_head_bbox(
                            single_pose_normalized_landmarks, frame_width, frame_height
                        )
                        if not current_pose_head_bbox:
                            continue

                        iou = self._calculate_iou(face_bbox_coords, current_pose_head_bbox)

                        if iou > best_iou_for_this_face: # Check if this pose is a better match for the current face
                            best_iou_for_this_face = iou
                            if iou >= self.face_to_pose_iou_threshold:
                                face_entry["full_body_pose_landmarks"] = [
                                    [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else (lm.presence if hasattr(lm, 'presence') else 1.0)] 
                                    for lm in single_pose_normalized_landmarks
                                ]
                                face_entry["matched_pose_head_bbox"] = current_pose_head_bbox
                                # Retrieve and store the segmentation mask
                                if pose_landmarker_result.segmentation_masks and pose_idx < len(pose_landmarker_result.segmentation_masks):
                                    mask_mp_image = pose_landmarker_result.segmentation_masks[pose_idx]
                                    face_entry["human_segmentation_mask"] = mask_mp_image.numpy_view()
                                else:
                                    face_entry["human_segmentation_mask"] = None # Should not happen if masks are enabled and indices align
                            else: # Best IoU so far, but below threshold for a definitive match
                                face_entry["full_body_pose_landmarks"] = None
                                face_entry["matched_pose_head_bbox"] = None
                                face_entry["human_segmentation_mask"] = None
                
                # Augment the specific face_entry dictionary with the best IOU found, even if no match.
                # The other fields (landmarks, head_bbox, mask) are set based on the best *above-threshold* match.
                face_entry["face_pose_iou"] = best_iou_for_this_face
        
        cap.release()
        print(f"PersonAssociator processed {processed_frames_count} frames for pose association.")
        return all_faces_data_by_frame_str # Return the (potentially) modified dictionary 