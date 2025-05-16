import torch
import numpy as np
from mmpose.apis import PoseEstimator as MMPoseModel # Using the high-level API
# from mmpose.structures import PoseDataSample # For understanding output, if needed

class MMPoseEstimator:
    def __init__(self, model_config_path: str, model_checkpoint_path: str, keypoint_convention: str = 'coco', device: str = 'cuda:0'):
        """
        Initializes the MMPoseEstimator.

        Args:
            model_config_path (str): Path to the MMPose model config file (e.g., 'hrnet_w48_coco_256x192.py').
            model_checkpoint_path (str): Path to the MMPose model checkpoint file or URL.
            keypoint_convention (str): The keypoint convention used by the model (e.g., 'coco', 'mpii').
            device (str): Device to run the model on (e.g., 'cuda:0' or 'cpu').
        """
        self.device = device
        self.estimator = MMPoseModel(
            model=model_config_path,
            weights=model_checkpoint_path,
            device=self.device
        )
        # You might need to know the keypoint convention (e.g., 'coco', 'mpii')
        # for get_pose_head_bbox. This often comes from the dataset used for training.
        self.keypoint_convention = keypoint_convention

    def estimate_poses(self, image_or_path, person_bboxes: list | None = None):
        """
        Performs pose estimation on a single image.

        Args:
            image_or_path (str | np.ndarray): Path to the image or the image as a NumPy array.
            person_bboxes (list, optional): A list of person bounding boxes [x1, y1, x2, y2]
                                            for top-down inference. If None, assumes the model
                                            is bottom-up or end-to-end.

        Returns:
            list: A list of PoseDataSample objects, each containing keypoints, scores, and bboxes
                  for a detected person. Returns an empty list if no poses are detected.
        """
        # The MMPose PoseEstimator can take a list of bboxes.
        # If person_bboxes are [N, 4], it expects them in that format.
        # If they are [N, 5] (with score), it should also work.
        # Ensure bboxes are in the correct format if provided.
        results_generator = self.estimator(image_or_path, bboxes=person_bboxes)
        
        # The estimator returns a generator. Consume it to get a list of PoseDataSample.
        pose_results = [result for result in results_generator]
        return pose_results

    @staticmethod
    def get_pose_head_bbox(keypoints: np.ndarray, keypoint_scores: np.ndarray, image_shape: tuple, 
                           keypoint_convention: str = 'coco', conf_threshold: float = 0.3,
                           padding_factor: float = 0.2) -> tuple | None:
        """
        Derives a head bounding box from pose keypoints.

        Args:
            keypoints (np.ndarray): Array of keypoints (N_kpts, 2) for a single person.
            keypoint_scores (np.ndarray): Array of scores (N_kpts,) for each keypoint.
            image_shape (tuple): Shape of the image (height, width) for clamping coordinates.
            keypoint_convention (str): The keypoint convention used (e.g., 'coco').
            conf_threshold (float): Minimum confidence for a keypoint to be used.
            padding_factor (float): Factor to pad the bounding box. 0.1 means 10% padding.

        Returns:
            tuple | None: (x_min, y_min, x_max, y_max) for the head, or None if not enough keypoints.
        """
        if keypoint_convention.lower() == 'coco':
            # COCO keypoints for head: 0:nose, 1:L_eye, 2:R_eye, 3:L_ear, 4:R_ear
            head_indices = [0, 1, 2, 3, 4]
        # Add other conventions like 'mpii' or 'posetrack' if needed
        # elif keypoint_convention.lower() == 'mpii':
        #     head_indices = [...] # Define for MPII
        else:
            # Fallback or raise error if convention unknown
            print(f"Warning: Unknown keypoint convention '{keypoint_convention}' for head bbox.")
            # Try to use the first 5 keypoints as a guess if available
            if len(keypoints) >= 5:
                head_indices = list(range(5))
            else:
                return None


        valid_head_kpts = []
        for idx in head_indices:
            if idx < len(keypoints) and keypoint_scores[idx] >= conf_threshold:
                valid_head_kpts.append(keypoints[idx])

        if not valid_head_kpts:
            return None

        valid_head_kpts = np.array(valid_head_kpts)
        x_min = np.min(valid_head_kpts[:, 0])
        y_min = np.min(valid_head_kpts[:, 1])
        x_max = np.max(valid_head_kpts[:, 0])
        y_max = np.max(valid_head_kpts[:, 1])

        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        x_min -= width * padding_factor
        y_min -= height * padding_factor
        x_max += width * padding_factor
        y_max += height * padding_factor


        # Clamp to image boundaries
        img_h, img_w = image_shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w -1, x_max)
        y_max = min(img_h -1, y_max)
        
        if x_min >= x_max or y_min >= y_max:
            return None

        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def close(self):
        """Clean up resources if any (e.g., if not using the high-level PoseEstimator which handles its own context)."""
        # For mmpose.apis.PoseEstimator, explicit cleanup is usually not needed
        # as it manages its own model lifecycle.
        # If we were using init_model directly, we might have things to clean.
        print("MMPoseEstimator closed (typically no specific action needed for mmpose.apis.PoseEstimator).") 