"""Module for tracking faces across video frames and selecting top frames."""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceTracker:
    """A class for tracking faces across video frames."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """Initialize the FaceTracker.

        Args:
            iou_threshold (float): Intersection over Union threshold for face tracking.
        """
        self.iou_threshold = iou_threshold

    def expand_box(self, box: List[float], expansion_ratio: float = 0.1) -> List[float]:
        """Expand the bounding box by a given ratio.

        Args:
            box (List[float]): Original bounding box coordinates [x1, y1, x2, y2].
            expansion_ratio (float): Ratio to expand the box.

        Returns:
            List[float]: Expanded bounding box coordinates.
        """
        width = box[2] - box[0]
        height = box[3] - box[1]

        x_expand = width * expansion_ratio
        y_expand = height * expansion_ratio

        expanded_box = [
            box[0] - x_expand,  # Left
            box[1] - y_expand,  # Top
            box[2] + x_expand,  # Right
            box[3] + y_expand,  # Bottom
        ]
        return expanded_box

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (List[float]): First bounding box coordinates.
            box2 (List[float]): Second bounding box coordinates.

        Returns:
            float: IoU value.
        """
        box1 = torch.tensor(box1, device=device, dtype=torch.float32)
        box2 = torch.tensor(box2, device=device, dtype=torch.float32)

        x1_inter = torch.max(box1[0], box2[0])
        y1_inter = torch.max(box1[1], box2[1])
        x2_inter = torch.min(box1[2], box2[2])
        y2_inter = torch.min(box1[3], box2[3])

        inter_area = torch.max(
            torch.tensor(0.0, device=device), x2_inter - x1_inter
        ) * torch.max(torch.tensor(0.0, device=device), y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou.item()

    def track_faces(
        self,
        face_data: List[Tuple[int, List[float], float]],
        min_faces_per_cluster: int,
    ) -> List[List[Dict[str, Any]]]:
        """Track faces within a scene based on IoU threshold."""
        clusters: List[List[Dict[str, Any]]] = []

        for frame_number, face, conf in face_data:
            face_added = False

            for cluster in clusters:
                last_face_in_cluster = cluster[-1]["face"]
                iou = self.calculate_iou(
                    self.expand_box(last_face_in_cluster), self.expand_box(face)
                )

                if iou > self.iou_threshold:
                    cluster.append({"frame": frame_number, "face": face, "conf": conf})
                    face_added = True
                    break

            if not face_added:
                clusters.append([{"frame": frame_number, "face": face, "conf": conf}])

        return [cluster for cluster in clusters if len(cluster) > min_faces_per_cluster]

    def track_faces_across_scenes(
        self, scene_data: pd.DataFrame, face_data: Dict[int, Dict[str, Any]]
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Track faces across all scenes in a video.

        Args:
            scene_data: DataFrame containing scene information.
            face_data: Face detection data for each frame.

        Returns:
            Dict[str, List[List[Dict[str, Any]]]]: Tracked faces across scenes.
        """
        all_tracked_faces = {}

        for index, row in tqdm(
            scene_data.iterrows(),
            total=scene_data.shape[0],
            desc="Tracking Faces Across Scenes",
        ):
            frame_start, frame_end = int(row["Start Frame"]), int(row["End Frame"])
            scene_id = f"scene_{index + 1}"

            n_frames = frame_end - frame_start + 1
            min_faces_per_cluster = min(max(n_frames // 2, 15), 30)  # 30 is FPS

            face_data_for_scene = []

            for i in range(frame_start, frame_end):
                faces = face_data[i]["detections"]
                if len(faces) != 0:
                    for f in faces:
                        face_data_for_scene.append((i, f["box"], f["confidence"]))

            # Skip scenes with no faces detected
            if not face_data_for_scene:
                continue

            tracked_faces = self.track_faces(face_data_for_scene, min_faces_per_cluster)
            all_tracked_faces[scene_id] = tracked_faces

        return all_tracked_faces


class FrameSelector:
    """A class for selecting top frames for each detected face."""

    def __init__(
        self,
        video_file: str,
        top_n: int = 3,
        output_dir: Optional[str] = None,
        save_images: bool = True,
    ) -> None:
        """Initialize the FrameSelector.

        Args:
            video_file (str): Path to the input video file.
            top_n (int): Number of top frames to select for each face.
            output_dir (Optional[str]): Directory to save output files.
            save_images (bool): Whether to save cropped face images.
        """
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate the brightness of an image using GPU if available.

        Args:
            image (np.ndarray): Input image.

        Returns:
            float: Brightness value.
        """
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return torch.mean(image_tensor).item()

    @staticmethod
    def calculate_blurriness(image: np.ndarray) -> float:
        """Calculate the blurriness of an image using GPU if available.

        Args:
            image (np.ndarray): Input image.

        Returns:
            float: Blurriness value.
        """
        laplacian = torch.tensor(cv2.Laplacian(image, cv2.CV_32F))
        return torch.var(laplacian).item()

    def save_cropped_face(
        self, face_image: np.ndarray, unique_face_id: str, frame_idx: int
    ) -> Optional[str]:
        """Save the cropped face image to disk and return the relative path.

        Args:
            face_image (np.ndarray): Cropped face image.
            unique_face_id (str): Unique identifier for the face.
            frame_idx (int): Frame index.

        Returns:
            Optional[str]: Relative path to the saved image, or None if not saved.
        """
        if self.output_dir and self.save_images:
            save_filename = f"{unique_face_id}_frame_{frame_idx}.jpg"
            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, face_image)
            return save_filename
        return None

    def select_top_frames_per_face(
        self, tracked_data: Dict[str, List[List[Dict[str, Any]]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Select top frames per face based on multiple criteria.

        Criteria include confidence, size, brightness, and blurriness.

        Args:
            tracked_data: Dictionary containing tracked face data.

        Returns:
            Dictionary of selected top frames for each face.
        """
        cap = cv2.VideoCapture(self.video_file)
        selected_frames: Dict[str, List[Dict[str, Any]]] = {}
        global_face_id = 0

        total_faces = sum(len(faces) for faces in tracked_data.values())

        with tqdm(total=total_faces, desc="Select Top Frames Per Face") as pbar:
            for scene_id, faces in tracked_data.items():
                selected_frames[scene_id] = []

                for face_id, face_group in enumerate(faces):
                    frame_scores = []

                    for entry in face_group:
                        frame_idx = entry["frame"]
                        face_coords = entry["face"]
                        confidence = entry["conf"]

                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(
                                f"Warning: Could not read frame {frame_idx}. Skipping."
                            )
                            continue

                        height, width, _ = frame.shape
                        x1, y1, x2, y2 = map(int, face_coords)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        width_cropped = max(0, x2 - x1)
                        height_cropped = max(0, y2 - y1)
                        if width_cropped == 0 or height_cropped == 0:
                            print(
                                f"Warning: Invalid bounding box {face_coords} "
                                f"for frame {frame_idx}. Skipping."
                            )
                            continue

                        face_image = frame[y1:y2, x1:x2]
                        if face_image.size == 0:
                            print(
                                f"Warning: Face image is empty for frame {frame_idx}. "
                                "Skipping."
                            )
                            continue

                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        face_size = width_cropped * height_cropped
                        brightness = self.calculate_brightness(gray_face)
                        blurriness = self.calculate_blurriness(gray_face)

                        # Normalize the components
                        normalized_face_size = face_size / (width * height)
                        normalized_brightness = brightness / 255.0
                        normalized_blurriness = blurriness / (
                            blurriness + 1e-6
                        )  # normalize blurriness itself

                        # Combine features into a score
                        score = (
                            confidence
                            + 0.5 * normalized_face_size
                            + 0.3 * normalized_brightness
                            - 0.2 * normalized_blurriness
                        )

                        # Save the image and get its relative path
                        relative_path = self.save_cropped_face(
                            face_image, f"{scene_id}_face_{face_id}", frame_idx
                        )

                        frame_scores.append(
                            {
                                "frame_idx": frame_idx,
                                "total_score": score,
                                "face_coord": face_coords,
                                "image_path": relative_path,  # Relative path in JSON
                            }
                        )

                    if frame_scores:
                        top_frames = sorted(
                            frame_scores, key=lambda x: x["total_score"], reverse=True
                        )[: self.top_n]

                        unique_face_id = f"{scene_id}_face_{face_id}"
                        global_unique_face_id = f"global_face_{global_face_id}"

                        selected_frames[scene_id].append(
                            {
                                "unique_face_id": unique_face_id,
                                "global_face_id": global_unique_face_id,
                                "top_frames": [
                                    {
                                        "frame_idx": frame["frame_idx"],
                                        "total_score": frame["total_score"],
                                        "face_coord": frame["face_coord"],
                                        "image_path": frame["image_path"],
                                    }
                                    for frame in top_frames
                                ],
                            }
                        )

                    global_face_id += 1
                    pbar.update(1)

        cap.release()
        return selected_frames
