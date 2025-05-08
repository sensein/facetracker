import cv2
import numpy as np
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import pandas as pd
import mediapipe as mp
# Imports for PoseLandmarker
from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python import vision as mp_vision
# Import the protobuf definition used elsewhere
from mediapipe.framework.formats import landmark_pb2

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.predict_num = 0
        self.additional_attributes = []
        self.confidence = 0.0
        self.latest_associated_data = None # To store landmarks or other data

    def update(self, bbox, associated_data=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if len(bbox) > 0:
            self.kf.update(self.convert_bbox_to_z(bbox))
            self.predict_num = 0
            self.latest_associated_data = associated_data # Store associated data
        else:
            self.predict_num += 1
            # Optionally clear associated_data if no detection, or let it persist
            # self.latest_associated_data = None 

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.get_state()

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, detections_associated_data: Optional[List[Any]] = None, additional_attributes=None, predict_num=5):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          img_size - [height, width] of the image
          detections_associated_data - Optional list of data (e.g. landmarks) for each detection in dets.
          additional_attributes - list of additional attributes for each detection (Optional)
        Returns:
          A numpy array of tracks in the format [[x1,y1,x2,y2,track_id,confidence],...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers.
        trks_bboxes = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = [] # Use list for intermediate results
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0] # Predict tracker state
            trks_bboxes[t, :] = pos[:4] # Store predicted bbox [x1, y1, x2, y2]
            trk.last_bbox = pos[:4] # Store the predicted bbox for association

            # Check for invalid states after prediction
            if np.any(np.isnan(pos)) or pos[2] < pos[0] or pos[3] < pos[1]:
                 to_del.append(t)
            # Also remove trackers that are out of bounds based on predicted position
            elif pos[0] >= img_size[1] or pos[1] >= img_size[0] or pos[2] <= 0 or pos[3] <= 0:
                 to_del.append(t)

        # Clean up trackers with invalid states or out of bounds predictions
        for t in reversed(to_del):
             self.trackers.pop(t)
             # Need to remove corresponding row from trks_bboxes if we use it later for association
             # It's safer to rebuild trks_bboxes after deletion or handle indices carefully.
             # Let's rebuild it for simplicity after popping.

        # Rebuild trks_bboxes after removing invalid trackers
        trks_bboxes = np.zeros((len(self.trackers), 4))
        for t, trk in enumerate(self.trackers):
             # Use the stored predicted bbox if available, otherwise re-predict (should be same)
             if hasattr(trk, 'last_bbox'):
                 trks_bboxes[t, :] = trk.last_bbox
             else: # Fallback if last_bbox wasn't set (e.g., first frame edge case)
                 trks_bboxes[t,:] = trk.get_state()[0][:4]

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks_bboxes, self.iou_threshold)

        # Update matched trackers with assigned detections
        for i, m in enumerate(matched):
            trk_idx = m[1]
            det_idx = m[0]
            trk = self.trackers[trk_idx]
            current_det_associated_data = detections_associated_data[det_idx] if detections_associated_data and det_idx < len(detections_associated_data) else None
            trk.update(dets[det_idx, :4], associated_data=current_det_associated_data) # Update with bbox and associated_data
            trk.confidence = dets[det_idx, 4] # Update confidence
            if additional_attributes:
                # Assuming additional_attributes length matches dets
                trk.additional_attributes.append(additional_attributes[det_idx])

        # Update non-matched trackers (predict only)
        # The predict step was already done for all trackers earlier
        # We just need to mark them as not updated with a detection
        # The KalmanBoxTracker's internal `time_since_update` handles this
        for t in unmatched_trks:
             # Optionally, we could explicitly call update with empty bbox if needed
             # self.trackers[t].update([]) # This increments predict_num in KalmanBoxTracker
             pass # Prediction already happened, time_since_update incremented in predict()

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4]) # Initialize with bbox
            current_det_associated_data = detections_associated_data[i] if detections_associated_data and i < len(detections_associated_data) else None
            trk.latest_associated_data = current_det_associated_data # Set initial associated data
            trk.confidence = dets[i, 4] # Set initial confidence
            if additional_attributes:
                trk.additional_attributes.append(additional_attributes[i])
            self.trackers.append(trk)

        # Filter out dead trackers and prepare output
        final_trackers = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # Retrieve the *updated* or *predicted* state for the current frame
            d = trk.get_state()[0][:4] # Get current bbox [x1, y1, x2, y2]

            # Filter based on age, hits, and validity
            # Condition: (Updated this frame OR Young and few frames passed) AND Valid state
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Add track to output list
                # Output format will be: [x1, y1, x2, y2, track_id, confidence, associated_data]
                # This makes the numpy array tricky if associated_data is complex. 
                # For now, let's return a list of dicts from Sort.update if associated_data is present.
                # Or, FaceTracker.track_faces can reconstruct this.
                # Let Sort.update return the standard array and a dict for associated data for active tracks.
                standard_track_data = np.concatenate((d, [trk.id, trk.confidence])).reshape(1, -1)
                final_trackers.append(standard_track_data)
                # We need a way to associate trk.latest_associated_data with this output.
                # Let's collect it in FaceTracker.track_faces based on trk.id

            i -= 1
            # Remove dead tracklet
            # Check age, prediction count, and basic bbox validity/bounds
            if (trk.time_since_update >= self.max_age or
                # trk.predict_num >= predict_num or # predict_num check removed, handled by max_age
                d[2] <= d[0] or d[3] <= d[1] or # Invalid bbox dimensions
                d[0] >= img_size[1] or d[1] >= img_size[0] or d[2] <= 0 or d[3] <= 0): # Out of bounds
                self.trackers.pop(i)

        if len(final_trackers) > 0:
            return np.concatenate(final_trackers) # Return numpy array (N, 6)
        return np.empty((0, 6)) # Return empty array with correct shape

    @staticmethod
    def associate_detections_to_trackers(detections, trackers_bboxes, iou_threshold = 0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        detections: numpy array of detections, shape (N, 5) -> [[x1,y1,x2,y2,score],...]
        trackers_bboxes: numpy array of predicted tracker bboxes, shape (M, 4) -> [[x1,y1,x2,y2],...]
        """
        if len(trackers_bboxes) == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,),dtype=int) # Correct empty unmatched_trackers shape
        if len(detections) == 0:
            return np.empty((0,2),dtype=int), np.empty((0,),dtype=int), np.arange(len(trackers_bboxes)) # Correct empty unmatched_detections shape

        # Calculate IoU matrix
        # Detections are rows, Trackers are columns
        iou_matrix = np.zeros((len(detections), len(trackers_bboxes)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk_bbox in enumerate(trackers_bboxes):
                # Pass only the bbox coordinates [x1, y1, x2, y2] to iou function
                iou_matrix[d,t] = Sort.iou(det[:4], trk_bbox) # Use static method iou

        # Use Hungarian algorithm (linear_sum_assignment) to find best matches
        # We want to maximize IoU, so we minimize negative IoU
        matched_indices = linear_sum_assignment(-iou_matrix)
        # Note: scipy.optimize.linear_sum_assignment returns row_ind, col_ind
        # matched_indices is tuple (array([row_indices]), array([col_indices]))

        unmatched_detections = []
        # Find detections that were not assigned a tracker
        all_det_indices = np.arange(len(detections))
        unmatched_detections = np.setdiff1d(all_det_indices, matched_indices[0], assume_unique=True).tolist()

        unmatched_trackers = []
        # Find trackers that were not assigned a detection
        all_trk_indices = np.arange(len(trackers_bboxes))
        unmatched_trackers = np.setdiff1d(all_trk_indices, matched_indices[1], assume_unique=True).tolist()

        # Filter matches based on IoU threshold
        matches = []
        for r, c in zip(matched_indices[0], matched_indices[1]):
            if iou_matrix[r, c] < iou_threshold:
                # If IoU is below threshold, treat them as unmatched
                unmatched_detections.append(r)
                unmatched_trackers.append(c)
            else:
                # Otherwise, it's a valid match
                matches.append([r, c]) # Store as [det_idx, trk_idx]

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches, dtype=int)

        # Convert lists to numpy arrays before returning
        return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)

    @staticmethod
    def iou(bb_test, bb_gt):
        # Ensure inputs are [x1, y1, x2, y2]
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        # Add small epsilon to avoid division by zero
        union = area_test + area_gt - wh + 1e-6
        o = wh / union
        return(o)

class FaceTracker:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.5):
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def track_faces(self, frame: int, face_data: List[Dict[str, Any]], img_size: tuple) -> List[Dict[str, Any]]:
        """
        Tracks faces using the SORT algorithm and includes associated landmark data.

        Args:
            frame (int): The current frame number.
            face_data (List[Dict[str, Any]]): List of detections for the current frame.
                Each dict should have 'bbox' ([x1, y1, x2, y2]), 'confidence',
                and 'landmarks'.
            img_size (tuple): The size of the image frame (height, width).

        Returns:
            List[Dict[str, Any]]: List of tracked faces for the current frame.
                Each dict contains 'id' (track_id), 'frame', 'bbox', 'confidence',
                and 'landmarks'.
        """
        detections_for_sort = []
        landmarks_for_detections = []
        if face_data:
            for face in face_data:
                detections_for_sort.append([*face['bbox'], face['confidence']])
                landmarks_for_detections.append(face.get('landmarks')) # Use .get for safety
            detections_np = np.array(detections_for_sort)
        else:
            detections_np = np.empty((0, 5))
            
        # Update SORT tracker, passing landmarks as associated data
        # Sort returns: [[x1, y1, x2, y2, track_id, confidence], ...]
        tracked_faces_array = self.sort_tracker.update(detections_np, img_size, detections_associated_data=landmarks_for_detections)
        
        result = []
        if tracked_faces_array.shape[0] > 0:
            # Create a mapping from sort_tracker's internal trk.id to its latest_associated_data
            tracker_id_to_landmarks = {trk.id: trk.latest_associated_data for trk in self.sort_tracker.trackers}

            for face_output in tracked_faces_array:
                track_id = int(face_output[4])
                face_info = {
                    "id": track_id,
                    "frame": frame,
                    "bbox": face_output[:4].tolist(),
                    "confidence": face_output[5],
                    "landmarks": tracker_id_to_landmarks.get(track_id) # Get landmarks using track_id
                }
                result.append(face_info)
        
        return result

class FrameSelector:
    # Define constants for head landmarks from MediaPipe Pose (BlazePose 33 landmarks)
    # These are: Nose, Left Eye (inner, center, outer), Right Eye (inner, center, outer), Left Ear, Right Ear, Mouth Left, Mouth Right
    # We'll use a bounding box derived from a wider set: eyes, ears, mouth, shoulders to be safer for head region.
    POSE_HEAD_LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Nose, Eyes, Ears, Mouth

    def __init__(
        self,
        video_file: str,
        top_n: int = 3,
        output_dir: Optional[str] = None,
        save_images: bool = True,
        pose_model_asset_path: str = "pose_landmarker_lite.task", # Placeholder
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_pose_tracking_confidence: float = 0.5, # For VIDEO mode, not used here
        face_to_pose_iou_threshold: float = 0.3,
    ):
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images
        self.face_to_pose_iou_threshold = face_to_pose_iou_threshold

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize PoseLandmarker
        try:
            pose_base_options = mp_python_tasks.BaseOptions(model_asset_path=pose_model_asset_path)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=5, # Max number of poses to detect in a frame
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_pose_tracking_confidence,
                output_segmentation_masks=False,
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            print("MediaPipe PoseLandmarker initialized successfully.")
        except Exception as e:
            print(f"Error initializing MediaPipe PoseLandmarker: {e}. Pose estimation will be skipped.")
            self.pose_landmarker = None

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        return cv2.mean(image)[0]

    @staticmethod
    def calculate_blurriness(image: np.ndarray) -> float:
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def save_cropped_face(
        self, face_image: np.ndarray, scene_id: str, track_id: int, frame_idx: int
    ) -> Optional[str]:
        if self.output_dir and self.save_images:
            # Use scene_id and track_id for a more unique filename
            save_filename = f"scene_{scene_id}_track_{track_id}_frame_{frame_idx}.jpg"
            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, face_image)
            return save_filename
        return None

    @staticmethod
    def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        Boxes are [x1, y1, x2, y2].
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
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _get_pose_head_bbox(
        self, 
        pose_landmarks: List[landmark_pb2.NormalizedLandmark], # Use landmark_pb2 type hint
        frame_width: int, 
        frame_height: int
    ) -> Optional[List[int]]:
        """Derive a head bounding box from pose landmarks."""
        if not pose_landmarks:
            return None
            
        head_x_coords = []
        head_y_coords = []
        
        for idx in self.POSE_HEAD_LANDMARK_INDICES:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                # Check landmark visibility before using it for bbox calculation
                if hasattr(landmark, 'visibility') and landmark.visibility < 0.7: # Visibility threshold
                    continue
                head_x_coords.append(landmark.x * frame_width)
                head_y_coords.append(landmark.y * frame_height)
        
        if not head_x_coords or not head_y_coords: # If no visible landmarks formed the list
            return None

        x1 = int(min(head_x_coords))
        y1 = int(min(head_y_coords))
        x2 = int(max(head_x_coords))
        y2 = int(max(head_y_coords))
        
        # Ensure valid box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)

        if x1 >= x2 or y1 >= y2:
            return None
            
        return [x1, y1, x2, y2]

    def _get_best_matching_pose_landmarks(
        self, 
        face_bbox: List[int], 
        pose_landmarker_result: mp_vision.PoseLandmarkerResult, 
        frame_width: int, 
        frame_height: int
    ) -> Optional[List[List[float]]]:
        """Find the body pose best matching the given face_bbox."""
        if not self.pose_landmarker or not pose_landmarker_result.pose_landmarks:
            return None

        best_iou = 0.0
        matched_pose_world_landmarks_xyzv = None # Store world landmarks if available, else normalized
        matched_pose_normalized_landmarks_xyzv = None


        for pose_idx, single_pose_normalized_landmarks in enumerate(pose_landmarker_result.pose_landmarks):
            # Type hint for iterated item can implicitly use the pb2 type if needed, but let's ensure clarity
            # The object itself might be a container, but we access attributes like pb2
            pose_head_bbox = self._get_pose_head_bbox(single_pose_normalized_landmarks, frame_width, frame_height)
            if not pose_head_bbox:
                continue

            iou = self._calculate_iou(face_bbox, pose_head_bbox)

            if iou > best_iou:
                best_iou = iou
                # Store all normalized landmarks for the best matching pose
                matched_pose_normalized_landmarks_xyzv = [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0] for lm in single_pose_normalized_landmarks]
                
                # If world landmarks are available (they usually are with PoseLandmarker)
                if pose_landmarker_result.pose_world_landmarks and pose_idx < len(pose_landmarker_result.pose_world_landmarks):
                    single_pose_world_landmarks = pose_landmarker_result.pose_world_landmarks[pose_idx]
                    matched_pose_world_landmarks_xyzv = [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0] for lm in single_pose_world_landmarks]


        if best_iou >= self.face_to_pose_iou_threshold:
            # Prioritize returning world landmarks if available, else normalized
            return matched_pose_world_landmarks_xyzv if matched_pose_world_landmarks_xyzv else matched_pose_normalized_landmarks_xyzv
        return None

    def select_top_frames_per_face(
        self, tracked_data_by_scene: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Selects the top N frames for each tracked face based on quality metrics.

        Args:
            tracked_data_by_scene (Dict[str, List[Dict[str, Any]]]): 
                A dictionary where keys are scene_ids and values are lists of face entries.
                Each face entry is a dict from FaceTracker, expected to contain:
                'id' (local track ID), 'frame' (frame_idx), 'bbox',
                'confidence', and 'landmarks' (pre-extracted from FaceDetector).
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: 
                Selected frames grouped by scene_id. Each list item contains:
                'unique_track_id': A string f"{scene_id}_{local_track_id}".
                'top_frames': List of dicts, each with 'frame_idx', 'total_score',
                              'face_coord', 'image_path', and 'face_mesh' (landmarks).
        """
        cap = cv2.VideoCapture(self.video_file)
        # Stores data for each unique track instance: (scene_id, local_track_id) -> list_of_frame_data
        unique_track_instances_data: Dict[tuple, List[Dict[str, Any]]] = {}

        total_face_entries = sum(len(faces) for faces in tracked_data_by_scene.values())

        with tqdm(total=total_face_entries, desc="Frame Selection & Pose Estimation") as pbar:
            for scene_id, scene_face_entries in tracked_data_by_scene.items():
                for face_entry in scene_face_entries:
                    local_track_id = face_entry["id"]
                    frame_idx = face_entry["frame"]
                    face_coords = face_entry["bbox"] 
                    confidence = face_entry["confidence"]
                    face_landmarks = face_entry.get("landmarks")

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx} for scene {scene_id}, track {local_track_id}. Skipping.")
                        pbar.update(1)
                        continue
                    
                    height, width = frame.shape[:2]
                    
                    # --- Full Body Pose Estimation ---
                    full_body_pose_data = None
                    if self.pose_landmarker:
                        try:
                            # Convert BGR frame to RGB then to MediaPipe Image
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_full_frame_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            pose_landmarker_result = self.pose_landmarker.detect(mp_full_frame_image)
                            
                            full_body_pose_data = self._get_best_matching_pose_landmarks(
                                face_coords, pose_landmarker_result, width, height
                            )
                        except Exception as e:
                            print(f"Error during pose estimation for frame {frame_idx}: {e}")
                    # --- End Full Body Pose Estimation ---

                    x1_f, y1_f, x2_f, y2_f = map(int, face_coords)
                    x1_f, y1_f = max(0, x1_f), max(0, y1_f)
                    x2_f, y2_f = min(width - 1, x2_f), min(height - 1, y2_f)

                    if x1_f >= x2_f or y1_f >= y2_f:
                        print(f"Warning: Invalid crop dimensions for scene {scene_id}, track {local_track_id}, frame {frame_idx}. Skipping.")
                        pbar.update(1)
                        continue
                        
                    face_image = frame[y1_f:y2_f, x1_f:x2_f]
                    if face_image.size == 0:
                        print(f"Warning: Empty face image for scene {scene_id}, track {local_track_id}, frame {frame_idx}. Skipping.")
                        pbar.update(1)
                        continue
                    
                    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_size = (x2_f - x1_f) * (y2_f - y1_f) # Use face crop dimensions
                    brightness = self.calculate_brightness(gray_face)
                    blurriness = self.calculate_blurriness(gray_face)

                    frame_area = width * height
                    normalized_face_size = face_size / frame_area if frame_area > 0 else 0
                    normalized_brightness = brightness / 255.0
                    normalized_blurriness = blurriness / (blurriness + 1e-6) if blurriness > 1e-6 else 0 

                    score = (
                        confidence
                        + 0.5 * normalized_face_size
                        + 0.3 * normalized_brightness
                        - 0.2 * normalized_blurriness 
                    )
                    
                    relative_path = self.save_cropped_face(face_image, scene_id, local_track_id, frame_idx)

                    unique_track_key = (scene_id, local_track_id)
                    if unique_track_key not in unique_track_instances_data:
                        unique_track_instances_data[unique_track_key] = []

                    unique_track_instances_data[unique_track_key].append({
                        "frame_idx": frame_idx,
                        "total_score": score,
                        "face_coord": face_coords,
                        "image_path": relative_path,
                        "face_mesh": face_landmarks, 
                        "full_body_pose": full_body_pose_data # Add full body pose
                    })
                    pbar.update(1)
        cap.release()

        selected_frames_output: Dict[str, List[Dict[str, Any]]] = {}
        for unique_track_key, frames_for_track in unique_track_instances_data.items():
            scene_id, local_track_id = unique_track_key
            top_frames_for_this_track = sorted(frames_for_track, key=lambda x: x["total_score"], reverse=True)[:self.top_n]
            
            if not top_frames_for_this_track:
                continue

            if scene_id not in selected_frames_output:
                selected_frames_output[scene_id] = []
            
            unique_track_id_str = f"{scene_id}_track_{local_track_id}" 

            selected_frames_output[scene_id].append({
                "unique_track_id": unique_track_id_str, 
                "top_frames": [
                    {
                        "frame_idx": f_data["frame_idx"],
                        "total_score": f_data["total_score"],
                        "face_coord": f_data["face_coord"],
                        "image_path": f_data["image_path"],
                        "face_mesh": f_data["face_mesh"],
                        "full_body_pose": f_data["full_body_pose"] # Include in output
                    }
                    for f_data in top_frames_for_this_track
                ]
            })
        return selected_frames_output

    def close(self):
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
            print("MediaPipe PoseLandmarker closed.")
        # Placeholder for other cleanup if needed
        print("FrameSelector closed.")
