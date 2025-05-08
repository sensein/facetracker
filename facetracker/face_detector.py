"""Face detection (MTCNN) and detailed landmark extraction (MediaPipe) module."""

# Imports (Combined and Corrected)
import os
import json
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN # Use MTCNN
import mediapipe as mp             # Keep for FaceLandmarker
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm
# Keep protobuf definition for type hints if needed, though might not be strictly necessary now
from mediapipe.framework.formats import landmark_pb2
from typing import Any, Dict, List, Tuple, Optional # Ensure Optional is imported

class FaceDetector:
    """
    Detects faces using MTCNN and extracts detailed landmarks using MediaPipe FaceLandmarker.
    """
    def __init__(
        self,
        video_path: str,
        output_dir: str, # Can be used for MTCNN annotated video or crops
        face_landmarker_model_path: str = "face_landmarker.task", # MP model
        device: str = "cuda" if torch.cuda.is_available() else "cpu", # For MTCNN
        mtcnn_keep_all: bool = True,
        mtcnn_min_face_size: int = 20,
        mtcnn_thresholds: list = [0.6, 0.7, 0.7],
        mtcnn_factor: float = 0.709,
        mtcnn_post_process: bool = True,
        mtcnn_min_confidence: float = 0.9, # Min confidence for MTCNN detection to process further
        mp_min_face_presence_confidence: float = 0.5, # MP Landmarker threshold
        padding_factor: float = 0.2 # Factor to expand MTCNN bbox for landmark detection crop
    ):
        """
        Initializes the FaceDetector with MTCNN and MediaPipe FaceLandmarker.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory for optional output files.
            face_landmarker_model_path (str): Path to the MediaPipe FaceLandmarker model bundle.
            device (str): Device for MTCNN ('cuda' or 'cpu').
            mtcnn_keep_all (bool): MTCNN keep_all flag.
            mtcnn_min_face_size (int): MTCNN min_face_size.
            mtcnn_thresholds (list): MTCNN thresholds.
            mtcnn_factor (float): MTCNN factor.
            mtcnn_post_process (bool): MTCNN post_process flag.
            mtcnn_min_confidence (float): Minimum confidence score from MTCNN to accept the detection.
            mp_min_face_presence_confidence (float): Min presence confidence for MP FaceLandmarker
                                                      on the cropped face image.
            padding_factor (float): How much to expand the MTCNN bounding box before
                                    feeding the crop to FaceLandmarker.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.device = device
        self.padding_factor = padding_factor
        self.mtcnn_min_confidence = mtcnn_min_confidence

        print(f"Initializing MTCNN on device: {self.device}")
        # Check if CUDA device is actually available if specified
        if "cuda" in self.device and not torch.cuda.is_available():
            print(f"Warning: CUDA specified but not available. Falling back to CPU for MTCNN.")
            self.device = "cpu"
            
        self.mtcnn = MTCNN(
            keep_all=mtcnn_keep_all,
            min_face_size=mtcnn_min_face_size,
            thresholds=mtcnn_thresholds,
            factor=mtcnn_factor,
            post_process=mtcnn_post_process,
            device=self.device
        )

        print("Initializing MediaPipe FaceLandmarker...")
        self.landmarker = None
        if face_landmarker_model_path and os.path.exists(face_landmarker_model_path):
            try:
                # Base options for FaceLandmarker
                base_options_lm = mp_python.BaseOptions(model_asset_path=face_landmarker_model_path)
                # Options specific to FaceLandmarker
                options_lm = vision.FaceLandmarkerOptions(
                    base_options=base_options_lm,
                    running_mode=vision.RunningMode.IMAGE, # Process crops as images
                    num_faces=1, # We process one crop at a time
                    min_face_presence_confidence=mp_min_face_presence_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options_lm)
                print("MediaPipe FaceLandmarker initialized.")
            except Exception as e:
                print(f"Warning: Error initializing MediaPipe FaceLandmarker: {e}. Detailed landmarks will not be available.")
                self.landmarker = None
        else:
            print(f"Warning: FaceLandmarker model path not provided or file not found ({face_landmarker_model_path}). Detailed landmarks will not be extracted.")


    def _transform_landmarks(self, landmarks_norm, crop_x1, crop_y1, crop_w, crop_h, frame_w, frame_h):
        """Transforms landmarks from crop-normalized to frame-normalized coordinates."""
        transformed_landmarks = []
        for lm in landmarks_norm:
            # Calculate absolute coordinates in the frame
            abs_x = lm.x * crop_w + crop_x1
            abs_y = lm.y * crop_h + crop_y1
            # Normalize absolute coordinates by the full frame dimensions
            frame_norm_x = abs_x / frame_w
            frame_norm_y = abs_y / frame_h
            frame_norm_z = lm.z # Z is relative, scale might differ, keep as is for now
            transformed_landmarks.append([frame_norm_x, frame_norm_y, frame_norm_z])
        return transformed_landmarks

    def _detect_faces_and_landmarks_frame(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect faces with MTCNN, then extract landmarks with FaceLandmarker on crops.

        Args:
            frame (np.ndarray): Input frame (BGR).

        Returns:
            List[Dict[str, Any]]: List of detected face data. Each dict contains:
                'bbox' (from MTCNN, [x1, y1, x2, y2]), 
                'confidence' (from MTCNN),
                'landmarks' (detailed 478 landmarks from FaceLandmarker, normalized to frame, or None).
        """
        faces_data = []
        frame_height, frame_width = frame.shape[:2]
        
        # --- Stage 1: MTCNN Detection --- 
        # MTCNN expects RGB PIL Image or numpy array. Convert frame.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Note: MTCNN can be slow, especially on CPU for large frames.
        # Consider resizing frame if performance is an issue, but adjust coordinates back.
        boxes, probs = self.mtcnn.detect(rgb_frame, landmarks=False) # landmarks=False is faster

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # Ensure prob is not None before comparison
                if prob is None or prob < self.mtcnn_min_confidence:
                    continue

                # MTCNN returns box as [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)

                # Ensure box coordinates are valid before padding/cropping
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
                    # print(f"Warning: Skipping invalid MTCNN box {box}")
                    continue

                # Add padding to the bounding box for the landmark crop
                box_w = x2 - x1
                box_h = y2 - y1
                pad_w = int(box_w * self.padding_factor)
                pad_h = int(box_h * self.padding_factor)

                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(frame_width, x2 + pad_w)
                crop_y2 = min(frame_height, y2 + pad_h)

                # Ensure crop coordinates are valid after padding
                if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                    # print(f"Warning: Skipping invalid padded crop area for box {box}")
                    continue

                face_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_crop_bgr.size == 0:
                    # print(f"Warning: Skipping empty face crop for box {box}")
                    continue # Skip if crop is empty

                # --- Stage 2: MediaPipe FaceLandmarker on Crop --- 
                detailed_landmarks_transformed = None
                if self.landmarker:
                    try:
                        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                        mp_image_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)
                        landmarker_result = self.landmarker.detect(mp_image_crop)

                        if landmarker_result and landmarker_result.face_landmarks:
                             # landmarks are normalized to the *crop*
                            landmarks_norm_crop = landmarker_result.face_landmarks[0] # num_faces=1
                            crop_w_actual = face_crop_rgb.shape[1]
                            crop_h_actual = face_crop_rgb.shape[0]
                            # Transform back to full frame normalized coordinates
                            detailed_landmarks_transformed = self._transform_landmarks(
                                landmarks_norm_crop, crop_x1, crop_y1, crop_w_actual, crop_h_actual, frame_width, frame_height
                            )
                    except Exception as e:
                        # print(f"Warning: FaceLandmarker failed on a crop: {e}")
                        pass # Continue without detailed landmarks if MP fails

                # Store result if MTCNN detection was good
                current_face_data = {
                    # Bbox from MTCNN [x1,y1,x2,y2] - ensure they are floats for consistency/JSON
                    "bbox": [float(b) for b in box],
                    "confidence": float(prob),      # Confidence from MTCNN
                    "landmarks": detailed_landmarks_transformed # Detailed 478 landmarks (normalized to frame) or None
                }
                faces_data.append(current_face_data)

        return faces_data


    def detect_faces_in_video(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect faces (MTCNN) and landmarks (MediaPipe) in the video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

        all_detections_by_frame_str = {}
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: # Handle case where total_frames is not available
            print("Warning: Could not determine total frames. Progress bar may be inaccurate.")
            total_frames = None

        pbar = tqdm(total=total_frames, desc="Detecting Faces (MTCNN+MP)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame using the two-stage method
            detected_faces_in_frame = self._detect_faces_and_landmarks_frame(frame)

            if detected_faces_in_frame:
                 # Use the correct key format expected by run_pipeline.py
                all_detections_by_frame_str[f"frame_{frame_count}"] = detected_faces_in_frame

            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        if total_frames:
             print(f"Completed detection. Found faces in {len(all_detections_by_frame_str)} / {total_frames} frames.")
        else:
             print(f"Completed detection. Found faces in {len(all_detections_by_frame_str)} frames (total frame count unknown).")
        return all_detections_by_frame_str

    def close(self):
         """Explicitly close resources."""
         if hasattr(self, 'landmarker') and self.landmarker:
             try:
                 self.landmarker.close()
                 print("MediaPipe FaceLandmarker closed.")
             except ValueError as e:
                # Ignore the specific error if the runner is already closed
                if "Task runner is currently not running" in str(e):
                    pass 
                else:
                    print(f"Ignoring unexpected ValueError closing FaceLandmarker in close: {e}")
             except Exception as e:
                 print(f"Ignoring generic error closing FaceLandmarker in close: {e}")
         # MTCNN doesn't typically require explicit closing

    def __del__(self):
        """Ensure resources are cleaned up."""
        # print("FaceDetector __del__ called") # Optional debug
        self.close()

    # _is_valid_box and save_results can be kept if needed, adapted to the new output format
    # They are not used by the main pipeline script directly but could be useful utilities.
