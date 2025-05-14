import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import os

# A utility function to draw landmarks, similar to what MediaPipe provides in solutions.drawing_utils
# This can be expanded or replaced by mediapipe.solutions.drawing_utils if preferred and compatible.
def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: mp_vision.PoseLandmarkerResult) -> np.ndarray:
    """Draws pose landmarks on an image.
    
    Args:
        rgb_image: The RGB image to draw landmarks on.
        detection_result: The PoseLandmarkerResult from MediaPipe.
        
    Returns:
        The image with landmarks drawn.
    """
    annotated_image = np.copy(rgb_image)
    if not detection_result or not detection_result.pose_landmarks:
        return annotated_image

    for pose_landmarks in detection_result.pose_landmarks:
        # Draw the landmarks
        for landmark in pose_landmarks:
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1) # Green dots for landmarks

        # Draw connections (example for a few connections, can be expanded using PoseLandmarker.POSE_CONNECTIONS)
        # This is a simplified version. For full connections, refer to MediaPipe's official examples.
        # Example: draw a line between nose (0) and left_eye_inner (1) if both are visible
        if len(pose_landmarks) > 1: # Ensure enough landmarks exist
            try:
                # A few example connections - for complete drawing, use mediapipe.solutions.drawing_utils
                # Define some connections if mediapipe.solutions.drawing_utils.POSE_CONNECTIONS is not directly used
                # For simplicity, we're just drawing landmarks here. Full connection drawing can be added.
                pass # Add connection drawing logic here if needed, or rely on landmarks only for now.
            except IndexError:
                print("Warning: Landmark index out of bounds during drawing connections.")
                pass
                
    return annotated_image


class PoseEstimator:
    """
    Estimates human poses in video frames using MediaPipe PoseLandmarker.
    """
    def __init__(
        self,
        model_asset_path: str = "pose_landmarker_lite.task",
        num_poses: int = 5, # Max number of poses to detect
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5, # For VIDEO mode, but good to have
        running_mode: mp_vision.RunningMode = mp_vision.RunningMode.VIDEO # Default to video for processing full videos
    ):
        """
        Initializes the PoseEstimator with MediaPipe PoseLandmarker.

        Args:
            model_asset_path (str): Path to the MediaPipe PoseLandmarker model bundle.
            num_poses (int): Maximum number of poses to detect.
            min_pose_detection_confidence (float): Minimum confidence for pose detection.
            min_pose_presence_confidence (float): Minimum confidence for pose presence.
            min_tracking_confidence (float): Minimum confidence for pose tracking (used in VIDEO mode).
            running_mode (mp_vision.RunningMode): MediaPipe running mode (IMAGE, VIDEO, LIVE_STREAM).
        """
        self.model_asset_path = model_asset_path
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.running_mode = running_mode
        self.landmarker = None
        self._initialize_landmarker()

    # POSE_HEAD_LANDMARK_INDICES: Nose, Eyes, Ears, Mouth
    # These indices are based on BlazePose (33 landmarks) as commonly used by MediaPipe PoseLandmarker.
    # 0: nose
    # 1: left_eye_inner, 2: left_eye, 3: left_eye_outer
    # 4: right_eye_inner, 5: right_eye, 6: right_eye_outer
    # 7: left_ear, 8: right_ear
    # 9: mouth_left, 10: mouth_right
    # (11: left_shoulder, 12: right_shoulder - could be used to expand for more robust head/upper body)
    POSE_HEAD_LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    @staticmethod
    def get_pose_head_bbox(\
        pose_landmarks: List[landmark_pb2.NormalizedLandmark], \
        frame_width: int, \
        frame_height: int,\
        visibility_threshold: float = 0.65\
    ) -> Optional[List[int]]:
        """Derive a head bounding box from pose landmarks.

        Args:
            pose_landmarks: A list of NormalizedLandmark objects for a single pose.
            frame_width: The width of the frame.
            frame_height: The height of the frame.
            visibility_threshold: Minimum visibility for a landmark to be included.

        Returns:
            A list [x1, y1, x2, y2] representing the bounding box, or None.
        """
        if not pose_landmarks:
            return None
            
        head_x_coords = []
        head_y_coords = []
        
        # Use the class attribute for indices
        for idx in PoseEstimator.POSE_HEAD_LANDMARK_INDICES:
            if idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                # Check landmark visibility/presence before using it
                # Use presence if available, fallback to visibility
                if hasattr(landmark, 'presence') and landmark.presence < visibility_threshold:
                    continue
                if hasattr(landmark, 'visibility') and landmark.visibility < visibility_threshold:
                    continue
                
                head_x_coords.append(landmark.x * frame_width)
                head_y_coords.append(landmark.y * frame_height)
        
        if not head_x_coords or not head_y_coords: # If no visible/present landmarks formed the list
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
            return None # Invalid box dimensions
            
        return [x1, y1, x2, y2]

    def _initialize_landmarker(self):
        """Helper to initialize the PoseLandmarker."""
        try:
            base_options = mp_python_tasks.BaseOptions(model_asset_path=self.model_asset_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=self.running_mode,
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.min_pose_detection_confidence,
                min_pose_presence_confidence=self.min_pose_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence, # Relevant for VIDEO mode
                output_segmentation_masks=True # Assuming we don't need masks for now
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            print("MediaPipe PoseLandmarker initialized successfully.")
        except Exception as e:
            print(f"Error initializing MediaPipe PoseLandmarker: {e}")
            self.landmarker = None
            raise # Reraise exception if initialization fails

    def detect_pose_in_frame(
        self, 
        frame_rgb: np.ndarray, 
        timestamp_ms: Optional[int] = None
    ) -> Optional[mp_vision.PoseLandmarkerResult]:
        """
        Detects poses in a single RGB frame.

        Args:
            frame_rgb (np.ndarray): The input frame in RGB format.
            timestamp_ms (Optional[int]): Timestamp for the frame in milliseconds, required for VIDEO mode.

        Returns:
            Optional[mp_vision.PoseLandmarkerResult]: The detection result, or None if error.
        """
        if not self.landmarker:
            print("PoseLandmarker not initialized.")
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            if self.running_mode == mp_vision.RunningMode.IMAGE:
                return self.landmarker.detect(mp_image)
            elif self.running_mode == mp_vision.RunningMode.VIDEO:
                if timestamp_ms is None:
                    raise ValueError("timestamp_ms is required for VIDEO running mode.")
                return self.landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                # LIVE_STREAM mode would require a callback and is handled differently.
                print(f"Running mode {self.running_mode} not directly supported by this method for synchronous detection.")
                return None
        except Exception as e:
            print(f"Error during pose detection in frame: {e}")
            return None

    def detect_poses_video(self, video_path: str) -> Dict[int, mp_vision.PoseLandmarkerResult]:
        """
        Detects poses in all frames of a video file.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            Dict[int, mp_vision.PoseLandmarkerResult]: A dictionary mapping frame index
                                                      to PoseLandmarkerResult.
        """
        if not self.landmarker:
            print("PoseLandmarker not initialized. Cannot process video.")
            return {}
        
        if self.running_mode != mp_vision.RunningMode.VIDEO and self.running_mode != mp_vision.RunningMode.IMAGE:
            print(f"Warning: Current running mode is {self.running_mode}. For video file processing, VIDEO or IMAGE mode is typical. Results might be unexpected.")
            # If mode is IMAGE, we can still process frame by frame. If it's LIVE_STREAM, this sync approach won't work.

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        all_pose_data: Dict[int, mp_vision.PoseLandmarkerResult] = {}
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pbar_desc = "Detecting Poses in Video"
        if self.running_mode == mp_vision.RunningMode.IMAGE:
            pbar_desc += " (IMAGE mode)"
        elif self.running_mode == mp_vision.RunningMode.VIDEO:
            pbar_desc += " (VIDEO mode)"

        with tqdm(total=total_frames, desc=pbar_desc) as pbar:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                pose_result = None
                if self.running_mode == mp_vision.RunningMode.IMAGE:
                    pose_result = self.detect_pose_in_frame(frame_rgb)
                elif self.running_mode == mp_vision.RunningMode.VIDEO:
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    pose_result = self.detect_pose_in_frame(frame_rgb, timestamp_ms=timestamp_ms)
                
                if pose_result:
                    all_pose_data[frame_idx] = pose_result
                
                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"Processed {frame_idx} frames for pose estimation.")
        return all_pose_data
        
    def close(self):
        """Closes the PoseLandmarker."""
        if self.landmarker:
            try:
                self.landmarker.close()
                print("MediaPipe PoseLandmarker closed.")
            except Exception as e:
                print(f"Error closing PoseLandmarker: {e}")
        self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def draw_landmarks(
        rgb_image: np.ndarray, 
        pose_landmarker_result: mp_vision.PoseLandmarkerResult,
        draw_connections: bool = True 
    ) -> np.ndarray:
        """
        Draws the pose landmarks and connections on an image.
        Uses mediapipe.solutions.drawing_utils for robust drawing.

        Args:
            rgb_image: The image on which to draw.
            pose_landmarker_result: The result from PoseLandmarker.
            draw_connections: Whether to draw connections between landmarks.

        Returns:
            The image with landmarks and connections drawn.
        """
        annotated_image = np.copy(rgb_image)
        if not pose_landmarker_result or not pose_landmarker_result.pose_landmarks:
            return annotated_image

        for pose_landmarks_list in pose_landmarker_result.pose_landmarks:
            # Convert NormalizedLandmarkList to the format expected by drawing_utils if necessary
            # For PoseLandmarker, pose_landmarks_list is already a list of NormalizedLandmark objects
            
            # Create a LandmarkList object for drawing_utils
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility if hasattr(landmark, 'visibility') else None)
                for landmark in pose_landmarks_list
            ])

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks_proto,
                connections=mp.solutions.pose.POSE_CONNECTIONS if draw_connections else None,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        return annotated_image 