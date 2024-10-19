"""Module for detecting and managing scene changes in video files."""

from typing import List, Optional, Tuple

import cv2
from scenedetect import SceneManager, VideoCaptureAdapter
from scenedetect.detectors import AdaptiveDetector, ContentDetector, HashDetector


class SceneDetector:
    """A class for detecting and managing scene changes in video files."""

    def __init__(
        self, video_path: str, detector_type: str = "adaptive", min_scene_len: int = 15
    ) -> None:
        """Initialize the SceneDetector.

        Args:
            video_path (str): Path to the input video file.
            detector_type (str): Type of scene detector to use
                ("adaptive", "content", or "hash").
            min_scene_len (int): Minimum length of a scene in frames.
        """
        self.video_path = video_path
        self.detector_type = detector_type.lower()
        self.min_scene_len = min_scene_len
        self.cap: Optional[cv2.VideoCapture] = None
        self.scene_manager: Optional[SceneManager] = None
        self.shots: Optional[List[Tuple[int, int, float, float]]] = None

    def initialize_scene_manager(self) -> None:
        """Initialize SceneManager with the chosen detector."""
        self.scene_manager = SceneManager()
        if self.detector_type == "content":
            self.scene_manager.add_detector(
                ContentDetector(min_scene_len=self.min_scene_len)
            )
        elif self.detector_type == "hash":
            self.scene_manager.add_detector(
                HashDetector(min_scene_len=self.min_scene_len)
            )
        else:  # Default to AdaptiveDetector
            self.scene_manager.add_detector(
                AdaptiveDetector(min_scene_len=self.min_scene_len)
            )

    def detect_scenes(self) -> None:
        """Detect scenes in the video file."""
        self.cap = cv2.VideoCapture(self.video_path)

        # Enable GPU acceleration in OpenCV (if available)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("GPU acceleration enabled.")
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        adapter = VideoCaptureAdapter(self.cap)
        if self.scene_manager is not None:
            self.scene_manager.detect_scenes(frame_source=adapter)
            self.shots = self.get_shot_list()
        self.cap.release()

    def get_shot_list(self) -> List[Tuple[int, int, float, float]]:
        """Calculate frame rate and convert frame numbers to seconds.

        Returns:
            List[Tuple[int, int, float, float]]: List of shots with frame numbers
                and timestamps.
        """
        if self.cap is None or self.scene_manager is None:
            return []
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        scene_list = self.scene_manager.get_scene_list()
        shots = [
            (
                int(scene[0].get_frames()),
                int(scene[1].get_frames()),
                scene[0].get_frames() / fps,
                scene[1].get_frames() / fps,
            )
            for scene in scene_list
        ]
        return shots

    def save_shots(self, output_file: str) -> None:
        """Save shot boundaries with both frame numbers and time in seconds.

        Args:
            output_file (str): Path to the output file.
        """
        if self.shots is None:
            return
        with open(output_file, "w") as f:
            f.write("Start Frame,End Frame,Start Time (s),End Time (s)\n")
            for start_frame, end_frame, start_time, end_time in self.shots:
                f.write(f"{start_frame},{end_frame},{start_time:.2f},{end_time:.2f}\n")

    def load_shots(self, input_file: str) -> None:
        """Load shot boundaries from a file."""
        self.shots = []
        with open(input_file, "r") as f:
            next(f)  # Skip the header line
            for line in f:
                values = line.strip().split(",")
                if len(values) == 4:
                    try:
                        self.shots.append(
                            (
                                int(float(values[0])),
                                int(float(values[1])),
                                float(values[2]),
                                float(values[3]),
                            )
                        )
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")

    def print_shots(self) -> None:
        """Print detected shots with both frame numbers and seconds."""
        if self.shots is None:
            return
        for i, (start_frame, end_frame, start_time, end_time) in enumerate(self.shots):
            print(
                f"Shot {i + 1}: Start Frame = {start_frame}, End Frame = {end_frame}, "
                f"Start Time = {start_time:.2f}s, End Time = {end_time:.2f}s"
            )
