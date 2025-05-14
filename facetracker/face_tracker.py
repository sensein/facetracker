import cv2
import numpy as np
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
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
    def __init__(self, max_age: int = 30, min_hits: int = 10, iou_threshold: float = 0.6):
        """
        max_age: Maximum number of frames to keep a track without associated detections.
        min_hits: Minimum number of detections before a track is considered valid.
        iou_threshold: IoU threshold for matching detections to existing tracks.
        """
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        print(f"FaceTracker initialized with Sort: max_age={max_age}, min_hits={min_hits}, iou_thresh={iou_threshold}")

    def track_faces(self, 
                    all_augmented_data_by_frame_str: Dict[str, List[Dict[str, Any]]], 
                    video_width: int, 
                    video_height: int) -> List[Dict[str, Any]]:
        """
        Tracks faces using the SORT algorithm across all provided frames.
        Assumes the entire video is processed as a single logical sequence or scene ("scene_0").

        Args:
            all_augmented_data_by_frame_str (Dict[str, List[Dict[str, Any]]]): 
                Data from PersonAssociator. Keys are frame_id strings (e.g., "frame_0", "frame_10"),
                values are lists of augmented face detection dictionaries for that frame.
                Each dict should have 'bbox', 'confidence', and other associated data like 
                'landmarks', 'full_body_pose_landmarks', etc.
            video_width (int): Width of the video frames.
            video_height (int): Height of the video frames.

        Returns:
            List[Dict[str, Any]]: A list of tracked face instances. Each instance dictionary contains:
                'id' (track_id), 'frame' (frame_idx), 'bbox', 'confidence', 
                and all other data carried from the input augmented_face_dict 
                (e.g., 'landmarks', 'full_body_pose_landmarks').
        """
        tracked_output_list = []
        img_size = (video_height, video_width)

        # Sort frame keys numerically for correct processing order
        # Assuming frame keys are like "frame_0", "frame_1", "frame_10", ...
        sorted_frame_keys = sorted(
            all_augmented_data_by_frame_str.keys(),
            key=lambda x: int(x.split('_')[-1])
        )
        
        print(f"FaceTracker processing {len(sorted_frame_keys)} frames...")

        for frame_key_str in tqdm(sorted_frame_keys, desc="Tracking faces frame-by-frame"):
            current_frame_idx = int(frame_key_str.split('_')[-1])
            augmented_faces_in_frame_list = all_augmented_data_by_frame_str.get(frame_key_str, [])

            detections_for_sort_np = []
            # associated_data_for_sort will be a list of the original augmented dicts
            associated_data_for_sort = []

            if augmented_faces_in_frame_list:
                for aug_face_dict in augmented_faces_in_frame_list:
                    # Ensure required keys are present, provide defaults if necessary
                    bbox = aug_face_dict.get('bbox')
                    confidence = aug_face_dict.get('confidence', 0.5) # Default confidence if missing

                    if bbox is None:
                        print(f"Warning: Missing 'bbox' in face data for frame {current_frame_idx}. Skipping this detection.")
                        continue
                    
                    detections_for_sort_np.append([*bbox, confidence])
                    associated_data_for_sort.append(aug_face_dict) # Pass the whole dict
            
            if not detections_for_sort_np: # No valid detections for sort in this frame
                detections_np = np.empty((0, 5))
            else:
                detections_np = np.array(detections_for_sort_np)

            # Update SORT tracker
            # Sort.update returns: [[x1, y1, x2, y2, track_id, confidence], ...]
            tracked_bboxes_array = self.sort_tracker.update(
                detections_np, 
                img_size, 
                detections_associated_data=associated_data_for_sort
            )
            
            if tracked_bboxes_array.shape[0] > 0:
                # Create a mapping from sort_tracker's internal trk.id to its latest_associated_data
                # This latest_associated_data is the complete aug_face_dict we passed in.
                tracker_id_to_full_data_map = {
                    trk.id: trk.latest_associated_data 
                    for trk in self.sort_tracker.trackers 
                    if trk.latest_associated_data is not None # Ensure data exists
                }

                for track_info_array in tracked_bboxes_array:
                    track_id = int(track_info_array[4])
                    
                    # Retrieve the full associated data dictionary for this track_id
                    full_associated_data = tracker_id_to_full_data_map.get(track_id)

                    if full_associated_data is None:
                        # This can happen if a tracker was predicted but not updated with a new detection
                        # in the current frame, but it's still considered active by Sort.
                        # Or if a new track was created from a detection for which we didn't have full_associated_data
                        # (though our logic above tries to ensure associated_data_for_sort mirrors detections_for_sort_np).
                        # For now, we'll skip if we can't find the rich data, or fill with basics.
                        # A more robust handling might involve carrying forward the *last known good data* for a track.
                        # print(f"Warning: No full associated data found for track_id {track_id} in frame {current_frame_idx}. Using basic info.")
                        # For now, we'll just use what SORT gives us and not add other fields.
                        # This means landmarks, pose, etc., might be missing for such instances.
                        # A better approach for Sort might be to ensure KalmanBoxTracker.latest_associated_data
                        # persists across predictions if not updated.
                        
                        # Let's construct a minimal entry if full_associated_data is missing
                        # but the track is valid according to Sort.
                        # The 'latest_associated_data' on the KalmanBoxTracker should hold the data from its *last update*.
                        
                        # Try to find the tracker directly to get its latest_associated_data
                        target_tracker = next((trk for trk in self.sort_tracker.trackers if trk.id == track_id), None)
                        if target_tracker and target_tracker.latest_associated_data:
                            full_associated_data = target_tracker.latest_associated_data
                        else:
                            # print(f"Debug: Still no data for track {track_id} in frame {current_frame_idx}")
                            # Fallback: create a very basic entry
                            tracked_instance = {
                                "id": track_id,
                                "frame": current_frame_idx,
                                "bbox": track_info_array[:4].tolist(),
                                "confidence": track_info_array[5],
                                # Other fields will be missing
                            }
                            tracked_output_list.append(tracked_instance)
                            continue # Skip to next track_info_array entry

                    # If full_associated_data was found (it should be the aug_face_dict)
                    tracked_instance = {
                        "id": track_id,
                        "frame": current_frame_idx, # Current frame index
                        "bbox": track_info_array[:4].tolist(), # Bbox from SORT
                        "confidence": track_info_array[5],   # Confidence from SORT (matches detection)
                        # Carry over all other data from the associated augmented face dictionary
                        **full_associated_data 
                    }
                    # Ensure 'bbox' and 'confidence' in tracked_instance are from SORT's output,
                    # as full_associated_data might have the original detection's bbox/confidence.
                    # The spread operator `**full_associated_data` might overwrite these if keys are same.
                    # Let's re-assign them to be sure.
                    tracked_instance['bbox'] = track_info_array[:4].tolist()
                    tracked_instance['confidence'] = track_info_array[5]
                    
                    tracked_output_list.append(tracked_instance)
        
        return tracked_output_list

class FrameSelector:
    # POSE_HEAD_LANDMARK_INDICES and related methods like _get_pose_head_bbox, _get_best_matching_pose_landmarks are removed.
    # _calculate_iou might be kept if it's used for other purposes, or moved/removed.
    # For now, let's assume _calculate_iou was only for pose matching and remove it too.

    def __init__(
        self,
        video_file: str,
        top_n: int = 3,
        output_dir: Optional[str] = None,
        save_images: bool = True,
        # Removed pose_model_asset_path and other pose-specific __init__ args
    ):
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images
        # Removed self.face_to_pose_iou_threshold

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Removed PoseLandmarker initialization
        # self.pose_landmarker = None 
        print("FrameSelector initialized (pose estimation is now external).")

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
            save_filename = f"scene_{scene_id}_track_{track_id}_frame_{frame_idx}.jpg"
            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, face_image)
            return save_filename
        return None

    def save_segmentation_outputs(
        self,
        full_frame_bgr: np.ndarray,
        segmentation_mask_float: np.ndarray,
        scene_id: str,
        track_id: int,
        frame_idx: int,
        mask_threshold: float = 0.5
    ) -> Dict[str, Optional[str]]:
        """Saves segmentation mask, segmented person, and isolated background."""
        output_paths = {
            "segmentation_mask_path": None,
            "segmented_person_path": None,
            "isolated_background_path": None,
        }
        if not self.output_dir or not self.save_images or segmentation_mask_float is None:
            return output_paths

        try:
            # 1. Threshold the float mask to get a binary mask (0 or 1)
            binary_mask_01 = (segmentation_mask_float > mask_threshold).astype(np.uint8)

            if binary_mask_01.ndim != 2:
                print(f"Warning: Segmentation mask for S:{scene_id}, T:{track_id}, F:{frame_idx} is not 2D. Skipping save.")
                return output_paths
            if full_frame_bgr.shape[:2] != binary_mask_01.shape[:2]:
                print(f"Warning: Frame and mask dimensions mismatch for S:{scene_id}, T:{track_id}, F:{frame_idx}. Resizing mask.")
                binary_mask_01 = cv2.resize(binary_mask_01, (full_frame_bgr.shape[1], full_frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask_01 = (binary_mask_01 > 0).astype(np.uint8) # Ensure still 0/1 after resize

            # Save the binary mask (as a 0/255 image)
            mask_filename = f"scene_{scene_id}_track_{track_id}_frame_{frame_idx}_human_mask.png"
            mask_save_path = os.path.join(self.output_dir, mask_filename)
            cv2.imwrite(mask_save_path, binary_mask_01 * 255)
            output_paths["segmentation_mask_path"] = mask_filename

            # 2. Segmented Person (person on black background)
            # The mask for bitwise_and should be single channel, 8-bit. binary_mask_01 is suitable.
            segmented_person = cv2.bitwise_and(full_frame_bgr, full_frame_bgr, mask=binary_mask_01)
            person_filename = f"scene_{scene_id}_track_{track_id}_frame_{frame_idx}_segmented_person.jpg"
            person_save_path = os.path.join(self.output_dir, person_filename)
            cv2.imwrite(person_save_path, segmented_person)
            output_paths["segmented_person_path"] = person_filename

            # 3. Isolated Background (background with person area blacked out)
            inverted_binary_mask_01 = 1 - binary_mask_01 # Person is 0, background is 1
            isolated_background = cv2.bitwise_and(full_frame_bgr, full_frame_bgr, mask=inverted_binary_mask_01)
            background_filename = f"scene_{scene_id}_track_{track_id}_frame_{frame_idx}_isolated_background.jpg"
            background_save_path = os.path.join(self.output_dir, background_filename)
            cv2.imwrite(background_save_path, isolated_background)
            output_paths["isolated_background_path"] = background_filename

        except Exception as e:
            print(f"Error saving segmentation outputs for S:{scene_id}, T:{track_id}, F:{frame_idx}: {e}")
        return output_paths

    def select_top_frames_per_face(
        self, tracked_data_by_scene: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Selects the top N frames for each tracked face based on quality metrics.
        Expects that 'full_body_pose_landmarks' (and optionally 'matched_pose_head_bbox', 'face_pose_iou')
        are already present in each face_entry if pose data is needed.
        """
        cap = cv2.VideoCapture(self.video_file)
        unique_track_instances_data: Dict[tuple, List[Dict[str, Any]]] = {}
        total_face_entries = sum(len(faces) for faces in tracked_data_by_scene.values())

        with tqdm(total=total_face_entries, desc="Frame Selection (Pose Data Pre-Associated)") as pbar:
            for scene_id, scene_face_entries in tracked_data_by_scene.items():
                for face_entry in scene_face_entries:
                    local_track_id = face_entry["id"]
                    frame_idx = face_entry["frame"]
                    face_coords = face_entry["bbox"]
                    confidence = face_entry["confidence"]
                    face_landmarks = face_entry.get("landmarks")
                    
                    # Retrieve pre-associated pose data
                    full_body_pose_data = face_entry.get("full_body_pose_landmarks")
                    # matched_pose_head_bbox = face_entry.get("matched_pose_head_bbox") # If needed by other logic
                    # face_pose_iou = face_entry.get("face_pose_iou") # If needed for scoring
                    human_segmentation_mask_data = face_entry.get("human_segmentation_mask")

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx} for scene {scene_id}, track {local_track_id}. Skipping.")
                        pbar.update(1)
                        continue
                    
                    height, width = frame.shape[:2]
                    
                    # --- Full Body Pose Estimation section is REMOVED ---

                    x1_f, y1_f, x2_f, y2_f = map(int, face_coords)
                    x1_f, y1_f = max(0, x1_f), max(0, y1_f)
                    x2_f, y2_f = min(width - 1, x2_f), min(height - 1, y2_f)

                    if x1_f >= x2_f or y1_f >= y2_f:
                        pbar.update(1); continue
                        
                    face_image = frame[y1_f:y2_f, x1_f:x2_f]
                    if face_image.size == 0:
                        pbar.update(1); continue
                    
                    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_size = (x2_f - x1_f) * (y2_f - y1_f)
                    brightness = self.calculate_brightness(gray_face)
                    blurriness = self.calculate_blurriness(gray_face)
                    frame_area = width * height
                    normalized_face_size = face_size / frame_area if frame_area > 0 else 0
                    normalized_brightness = brightness / 255.0
                    # Ensure blurriness does not lead to division by zero or negative scores if too low
                    normalized_blurriness = blurriness / (blurriness + 1e-6) if blurriness > 1e-6 else 0 

                    score = (
                        confidence
                        + 0.5 * normalized_face_size
                        + 0.3 * normalized_brightness
                        - 0.2 * normalized_blurriness 
                        # Consider if face_pose_iou or presence of full_body_pose_data should affect the score
                        # For example: if full_body_pose_data: score += 0.1 
                    )
                    
                    relative_path = self.save_cropped_face(face_image, scene_id, local_track_id, frame_idx)
                    
                    segmentation_outputs_paths = {} # Initialize
                    if human_segmentation_mask_data is not None and self.save_images:
                        # 'frame' is the full BGR frame from cap.read()
                        segmentation_outputs_paths = self.save_segmentation_outputs(
                            frame, human_segmentation_mask_data, scene_id, local_track_id, frame_idx
                        )
                        
                    unique_track_key = (scene_id, local_track_id)
                    if unique_track_key not in unique_track_instances_data:
                        unique_track_instances_data[unique_track_key] = []

                    frame_data_for_track = {
                        "frame_idx": frame_idx,
                        "total_score": score,
                        "face_coord": face_coords,
                        "image_path": relative_path, # Cropped face path
                        "face_mesh": face_landmarks,
                        "full_body_pose": full_body_pose_data, # This is now passed through
                        "segmentation_mask_path": segmentation_outputs_paths.get("segmentation_mask_path"),
                        "segmented_person_path": segmentation_outputs_paths.get("segmented_person_path"),
                        "isolated_background_path": segmentation_outputs_paths.get("isolated_background_path"),
                        # Optionally pass the raw mask data if needed downstream, though it can be large.
                        # "raw_human_segmentation_mask": human_segmentation_mask_data
                    }
                    unique_track_instances_data[unique_track_key].append(frame_data_for_track)
                    pbar.update(1)
        cap.release()

        selected_frames_output: Dict[str, List[Dict[str, Any]]] = {}
        for unique_track_key, frames_for_track in unique_track_instances_data.items():
            scene_id, local_track_id = unique_track_key
            top_frames_for_this_track = sorted(frames_for_track, key=lambda x: x["total_score"], reverse=True)[:self.top_n]
            if not top_frames_for_this_track: continue
            if scene_id not in selected_frames_output: selected_frames_output[scene_id] = []
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
                        "full_body_pose": f_data["full_body_pose"],
                        "segmentation_mask_path": f_data.get("segmentation_mask_path"),
                        "segmented_person_path": f_data.get("segmented_person_path"),
                        "isolated_background_path": f_data.get("isolated_background_path")
                    }
                    for f_data in top_frames_for_this_track
                ]
            })
        return selected_frames_output

    def close(self):
        # Removed self.pose_landmarker.close()
        print("FrameSelector closed (no internal pose landmarker to close).")
