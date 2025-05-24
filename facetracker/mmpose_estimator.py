"""MMPose-based pose estimation module."""
import cv2
import numpy as np
# from mmpose.apis import PoseEstimator as MMPoseModel # Old import
from mmpose.apis import MMPoseInferencer # New import for MMPose v1.x
from typing import List, Dict, Any, Tuple, Optional
import os

class MMPoseEstimator:
    """
    Estimates human pose using a specified MMPose model via MMPoseInferencer.
    Handles initialization of the model and processes images/frames for pose keypoints.
    """
    def __init__(
        self,
        pose_model_config: str,
        pose_model_checkpoint: str,
        device: str = 'cuda:0',
        keypoint_convention: str = 'coco', # e.g., 'coco', 'animalpose', etc.
        detector_device: str = None  # Allow separate device for detector
    ):
        """
        Initializes the MMPoseEstimator with a specific model configuration and checkpoint.

        Args:
            pose_model_config (str): Path to the MMPose model's config file.
            pose_model_checkpoint (str): Path to the MMPose model's checkpoint file.
            device (str): Device to run the model on (e.g., 'cuda:0' or 'cpu').
            keypoint_convention (str): The convention of the keypoints (e.g., 'coco').
                                       This helps in interpreting the output keypoints.
            detector_device (str): Device for detector (if different from main device). 
                                  If None, uses same as device.
        """
        print(f"Initializing MMPoseInferencer with config: {pose_model_config} and checkpoint: {pose_model_checkpoint} on device: {device}")
        
        # If CUDA device specified but we want to handle compatibility issues
        effective_device = device
        if detector_device is not None:
            print(f"Using separate detector device: {detector_device}")
        elif "cuda" in device:
            # Check if we should fallback due to CUDA compatibility issues
            try:
                import torch
                if torch.cuda.is_available():
                    # Test basic CUDA operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    test_result = test_tensor + 1
                    print(f"CUDA test passed on {device}")
                else:
                    print(f"CUDA not available, falling back to CPU")
                    effective_device = "cpu"
            except Exception as e:
                print(f"CUDA test failed: {e}, falling back to CPU")
                effective_device = "cpu"
        
        # Set environment variable to potentially help with CUDA compatibility
        if "cuda" in effective_device:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        try:
            self.inferencer = MMPoseInferencer(
                pose2d=pose_model_config, 
                pose2d_weights=pose_model_checkpoint, 
                device=effective_device  # Use the effective device for the inferencer
            )
        except Exception as e:
            if "cuda" in effective_device and "cuda" not in str(e).lower():
                print(f"Failed to initialize with {effective_device}, trying CPU: {e}")
                effective_device = "cpu"
                self.inferencer = MMPoseInferencer(
                    pose2d=self.pose_model_config,  # Use stored original path
                    pose2d_weights=self.pose_model_checkpoint,  # Use stored original path
                    device="cpu"
                )
            else:
                raise e
            
        self.keypoint_convention = keypoint_convention
        self.device = effective_device
        # Store original paths for potential CPU fallback
        self.pose_model_config = pose_model_config
        self.pose_model_checkpoint = pose_model_checkpoint
        # Input_size can often be inferred or is handled by the model's config, 
        # but if specific preprocessing is needed, it might be relevant.
        # For now, we rely on MMPoseInferencer's internal handling.
        print(f"MMPoseInferencer initialized for {keypoint_convention} convention on {effective_device}.")

    def estimate_poses(
        self, 
        frame: np.ndarray,
        bboxes: Optional[List[List[float]]] = None # Bounding boxes [x1, y1, x2, y2] for specific RoIs
    ) -> List[Dict[str, Any]]:
        """
        Estimates poses in the given frame.
        If bboxes are provided, estimates pose for each bounding box.
        Otherwise, performs person detection first (if model supports it) or whole-image pose estimation.

        Args:
            frame (np.ndarray): The input image/frame (BGR format from OpenCV).
            bboxes (Optional[List[List[float]]]): A list of bounding boxes [[x1, y1, x2, y2], ...]. 
                                                 If None, MMPoseInferencer may perform its own detection 
                                                 or process the whole image based on its configuration.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
                'keypoints': an array of keypoints (x, y) for a detected person.
                'keypoint_scores': an array of confidence scores for each keypoint.
                'bbox': an array [x1, y1, x2, y2, score] for the person, if provided/detected.
                      Note: MMPoseInferencer might return bboxes slightly differently or not at all
                      if input bboxes are already provided. We'll try to standardize.
        """
        # MMPoseInferencer expects BGR images by default if using OpenCV backend for loading, 
        # or RGB if using other backends. Since we pass a numpy array, it's good practice
        # to ensure it's in the format the model expects (often RGB).
        # However, many MMPose models handle BGR from OpenCV directly.
        # For now, let's assume BGR is fine as it's common with cv2.imread.
        # If issues arise, convert: frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MMPoseInferencer call: input can be image path, URL, or numpy array.
        # It returns a generator, so we convert it to a list.
        # `bboxes` argument for the inferencer itself might be named differently or handled
        # as part of the input dict. Let's check its API.
        # According to MMPoseInferencer docs, if bboxes are provided, they should be passed as a list of lists/arrays.
        
        try:
            results_generator = self.inferencer(frame, bboxes=bboxes, return_vis=False, show=False)
            results = list(results_generator)
        except RuntimeError as e:
            if "CUDA error" in str(e) and "cuda" in self.device:
                print(f"CUDA error encountered: {e}")
                print("Attempting to reinitialize MMPoseInferencer on CPU...")
                
                # Reinitialize on CPU
                try:
                    # Store original device
                    original_device = self.device
                    self.device = "cpu"
                    
                    # Get the config and checkpoint from the original inferencer
                    from mmpose.apis import MMPoseInferencer
                    
                    # Reinitialize with CPU
                    self.inferencer = MMPoseInferencer(
                        pose2d=self.pose_model_config,  # Use stored original path
                        pose2d_weights=self.pose_model_checkpoint,  # Use stored original path
                        device="cpu"
                    )
                    
                    print(f"Successfully reinitialized MMPoseInferencer on CPU")
                    
                    # Retry the inference
                    results_generator = self.inferencer(frame, bboxes=bboxes, return_vis=False, show=False)
                    results = list(results_generator)
                    
                except Exception as cpu_e:
                    print(f"Failed to reinitialize on CPU: {cpu_e}")
                    print("Returning empty results for this frame")
                    return []
            else:
                raise e

        processed_poses = []
        if results and 'predictions' in results[0]:
            for instance_predictions in results[0]['predictions']:
                # Each `instance_predictions` is typically a list for each detected/provided person/instance
                for pred in instance_predictions: # If multiple instances were processed from bboxes
                    keypoints = pred.get('keypoints', [])
                    keypoint_scores = pred.get('keypoint_scores', [])
                    # Inferencer might provide 'bbox' directly. If we passed bboxes in,
                    # we might want to associate results back to those original bboxes.
                    # The 'bbox' from pred is usually [x1, y1, x2, y2, score].
                    bbox_pred = pred.get('bbox', [0,0,0,0,0]) 
                    if len(bbox_pred) == 4: # if score is not included
                        bbox_pred.append(1.0) # Assume score 1.0 if not present

                    processed_poses.append({
                        'keypoints': np.array(keypoints, dtype=np.float32),
                        'keypoint_scores': np.array(keypoint_scores, dtype=np.float32),
                        'bbox': np.array(bbox_pred, dtype=np.float32) # [x1, y1, x2, y2, score]
                    })
        
        return processed_poses

    def get_keypoint_convention(self) -> str:
        """Returns the keypoint convention used by the model."""
        return self.keypoint_convention

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

# Example usage (for testing purposes, if run directly)
if __name__ == '__main__':
    # This example assumes you have a COCO model config and checkpoint.
    # You'll need to download them or use paths to your local models.
    # Example paths (replace with actual paths):
    # MODEL_CONFIG = 'mmpose_models/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    # MODEL_CHECKPOINT = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    
    # For a quick test, you might need to provide actual model files.
    # Let's assume the user has a model like 'rtmpose-m_8xb256-420e_coco-256x192.py'
    # and its corresponding .pth file in a 'mmpose_models' directory.
    
    # A more robust example would require actual model files.
    # For now, we'll just print a message if this is run directly.
    print("MMPoseEstimator class definition. To test, instantiate with model paths and call estimate_poses.")
    print("Example: estimator = MMPoseEstimator(pose_model_config='path/to/config.py', pose_model_checkpoint='path/to/checkpoint.pth')")

    # Dummy test (requires model files to actually run)
    # try:
    #     # Ensure you have these files or use valid downloadable URLs for MMPoseInferencer
    #     estimator = MMPoseEstimator(
    #         pose_model_config='td-hm_hrnet-w32_8xb64-210e_coco-256x192.py', # Placeholder, needs actual file or alias
    #         pose_model_checkpoint='hrnet_w32_coco_256x192-c78dce93_20200708.pth', # Placeholder
    #         device='cpu'
    #     )
    #     # Create a dummy image
    #     dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    #     cv2.rectangle(dummy_frame, (100, 100), (200, 300), (0, 255, 0), 2) # Draw a dummy person bbox
    #     dummy_bboxes = [[100, 100, 200, 300]]

    #     poses = estimator.estimate_poses(dummy_frame, bboxes=dummy_bboxes)
    #     if poses:
    #         print(f"Estimated {len(poses)} poses.")
    #         print(f"First pose keypoints: {poses[0]['keypoints']}")
    #     else:
    #         print("No poses estimated.")
    # except Exception as e:
    #     print(f"Could not run example: {e}. Ensure model config/checkpoint paths are correct and MMPose is installed.") 