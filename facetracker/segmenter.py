import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch
import os
import cv2

class ImageSegmenter:
    def __init__(self, 
                 model_type: str = "hiera_l", 
                 checkpoint_path: Optional[str] = None,
                 model_config_path: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initializes the ImageSegmenter with a SAM2 model.

        Args:
            model_type (str): Type of SAM2 model to load (e.g., "hiera_l", "hiera_b+").
            checkpoint_path (Optional[str]): Path to the SAM2 model checkpoint (.pth file).
            model_config_path (Optional[str]): Path to the SAM2 model configuration (.yaml file).
            device (str): Device to load the model on ("cuda" or "cpu").
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model_config_path = model_config_path
        self.device = device
        
        self.predictor: Optional[SAM2ImagePredictor] = None

        print(f"ImageSegmenter initialized with model type {self.model_type} on {self.device}.")
        
        if self.checkpoint_path and self.model_config_path:
            print(f"Loading SAM2 model from checkpoint: {self.checkpoint_path} and config: {self.model_config_path}")
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")
            if not os.path.exists(self.model_config_path):
                raise FileNotFoundError(f"Model config file not found at {self.model_config_path}")

            try:
                if self.device == "cuda" and not torch.cuda.is_available():
                    print("Warning: CUDA selected but not available. Falling back to CPU.")
                    self.device = "cpu"
                
                sam_model = build_sam2(self.model_config_path, self.checkpoint_path)
                sam_model.to(self.device)
                sam_model.eval()

                self.predictor = SAM2ImagePredictor(sam_model)
                print(f"SAM2 model '{self.model_type}' loaded successfully on device '{self.device}'.")

            except Exception as e:
                raise RuntimeError(f"Error loading SAM2 model: {e}")
        else:
            print("Warning: Checkpoint path or model config path not provided. SAM2 model not loaded.")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image for SAM2.
        
        Args:
            image (np.ndarray): Input image in BGR or RGB format.
            
        Returns:
            np.ndarray: Preprocessed image in RGB format.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Convert to RGB if needed
        if image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if image.shape[2] == 3 and image[0,0,0] > image[0,0,2]:  # Simple BGR check
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input image must be RGB or BGR with 3 channels")
            
        return image

    def segment_all(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments all possible objects in the given image using SAM2's automatic mask generation.
        
        Args:
            image (np.ndarray): Input image in BGR or RGB format.
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing segmentation masks and metadata.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Cannot perform segmentation.")
        
        try:
            image = self._preprocess_image(image)
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                self.predictor.set_image(image)
                # Use SAM2's automatic mask generation
                masks = self.predictor.generate()
                
                # Process and return the masks with metadata
                results = []
                for mask_data in masks:
                    results.append({
                        'mask': mask_data['segmentation'],
                        'score': mask_data.get('stability_score', 0.0),
                        'bbox': mask_data.get('bbox', None),
                        'area': mask_data.get('area', 0)
                    })
                return results
                
        except Exception as e:
            print(f"Error during automatic segmentation: {e}")
            return []

    def segment_human(self, image: np.ndarray, face_bbox: Optional[List[float]] = None, pose_keypoints: Optional[List[List[float]]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segments a human in the image using face bounding box or pose keypoints as prompts.
        
        Args:
            image (np.ndarray): Input image in BGR or RGB format.
            face_bbox (Optional[List[float]]): Face bounding box [x_min, y_min, x_max, y_max].
            pose_keypoints (Optional[List[List[float]]]): List of pose keypoints [[x,y],...].
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Human mask and background mask.
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Cannot segment human.")

        try:
            image = self._preprocess_image(image)
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                self.predictor.set_image(image)

                input_points_np: Optional[np.ndarray] = None
                input_labels_np: Optional[np.ndarray] = None
                input_box_np: Optional[np.ndarray] = None

                if face_bbox:
                    input_box_np = np.array([face_bbox], dtype=np.float32)
                
                if pose_keypoints:
                    points = [[kp[0], kp[1]] for kp in pose_keypoints]
                    input_points_np = np.array(points, dtype=np.float32)
                    input_labels_np = np.ones(input_points_np.shape[0], dtype=np.int32)

                if input_box_np is None and input_points_np is None:
                    print("No prompts provided. Using automatic segmentation and selecting largest mask.")
                    # Try automatic segmentation and select the largest mask
                    masks = self.segment_all(image)
                    if not masks:
                        return None, None
                    # Select the mask with the largest area
                    largest_mask = max(masks, key=lambda x: x['area'])
                    human_mask = largest_mask['mask']
                else:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=input_points_np,
                        point_labels=input_labels_np,
                        box=input_box_np,
                        multimask_output=False
                    )
                    
                    if masks is None or len(masks) == 0:
                        return None, None
                    human_mask = masks[0]

                human_mask_bool = human_mask.astype(bool)
                background_mask = ~human_mask_bool
                
                return human_mask_bool, background_mask

        except Exception as e:
            print(f"Error during human segmentation: {e}")
            return None, None

    def segment_frame_with_prompts(
        self, 
        frame_bgr: np.ndarray, 
        bboxes_abs: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        """
        Segments objects in a frame using bounding box prompts.
        
        Args:
            frame_bgr (np.ndarray): Input BGR frame.
            bboxes_abs (Optional[List[Tuple[int, int, int, int]]]): List of bboxes [[x1,y1,x2,y2],...].
            
        Returns:
            Tuple[Optional[List[np.ndarray]], Optional[List[float]]]: Masks and scores.
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Cannot segment frame.")

        try:
            frame_rgb = self._preprocess_image(frame_bgr)
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                self.predictor.set_image(frame_rgb)
                
                masks_out, scores_out = [], []
                if bboxes_abs:
                    for box in bboxes_abs:
                        input_box = np.array([box], dtype=np.float32)
                        masks, scores, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box,
                            multimask_output=False
                        )
                        if masks is not None and len(masks) > 0:
                            masks_out.append(masks[0])
                            scores_out.append(scores[0])
                
                return masks_out if masks_out else None, scores_out if scores_out else None

        except Exception as e:
            print(f"Error during frame segmentation: {e}")
            return None, None

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'predictor') and self.predictor is not None:
            del self.predictor
            self.predictor = None
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}")

    def __del__(self):
        self.close() 