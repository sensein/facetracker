"""Face detection module for video processing."""

import json
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    """A class for detecting and annotating faces in video frames."""

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_confidence: float = 0.8,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save output files.
            device (str): Device to use for computation ('cuda' or 'cpu').
            min_confidence (float): Minimum confidence threshold for face detection.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True, device=self.device, factor=0.6
        )  # Use MTCNN for face detection with scaling factor
        self.min_confidence = min_confidence

    def detect_faces_in_video(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect faces in the video and return face detections.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing face
            detections for each frame.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():  # Check if the video file was opened successfully
            raise ValueError(f"Error opening video file: {self.video_path}")

        face_detections = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, annotated_frame = self._detect_and_annotate_frame(frame)

            if faces:
                face_detections[f"frame_{frame_count}"] = faces

            cv2.imwrite(f"{self.output_dir}/frame_{frame_count}.jpg", annotated_frame)
            frame_count += 1

        cap.release()
        return face_detections

    def _detect_and_annotate_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect faces in a frame and annotate it.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray]: Detected faces and
            annotated frame.
        """
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > self.min_confidence and self._is_valid_box(box):
                    x1, y1, x2, y2 = map(int, box)
                    faces.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(prob),
                        }
                    )

        annotated_frame = self._annotate_frame(frame, faces)
        return faces, annotated_frame

    def _annotate_frame(
        self,
        frame: np.ndarray,
        faces: List[Dict[str, Any]],
        rect_color: Tuple[int, int, int] = (0, 255, 0),
        circle_color: Tuple[int, int, int] = (0, 0, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Annotate the frame with detected faces."""
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, circle_color, -1)

            confidence = face["confidence"]
            label = f"Face: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                text_color,
                2,
            )

        return frame

    def _is_valid_box(
        self, box: List[float], min_size: int = 20, max_aspect_ratio: float = 1.5
    ) -> bool:
        """Check if the bounding box is valid."""
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = max(width, height) / min(width, height)
        return (
            width >= min_size
            and height >= min_size
            and aspect_ratio <= max_aspect_ratio
        )

    def save_results(
        self, output_file: str, face_detections: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Save face detection results to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(face_detections, f, indent=4)
