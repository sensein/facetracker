import pytest
import numpy as np
from facetracker.segmenter import ImageSegmenter
import cv2 # Added for video processing
import os  # Added for path manipulation

# Use a smaller image size for testing to speed things up
DUMMY_IMAGE_HEIGHT = 60
DUMMY_IMAGE_WIDTH = 80

@pytest.fixture
def dummy_image():
    """Provides a small dummy RGB image for testing."""
    return np.random.randint(0, 256, 
                             (DUMMY_IMAGE_HEIGHT, DUMMY_IMAGE_WIDTH, 3), 
                             dtype=np.uint8)

@pytest.fixture
def segmenter_instance():
    """Provides an instance of ImageSegmenter (without loading the actual model)."""
    # We instantiate it but don't call _load_model (which is commented out anyway)
    # This allows testing the interface and placeholder logic
    segmenter = ImageSegmenter(model_type="hiera_l", checkpoint_path=None, device="cpu")
    # Manually set dummy predictor/generator to bypass the "not initialized" checks
    # In a real test setup with loaded models, these would be set by _load_model
    segmenter.predictor = True # Just needs to be non-None for the check
    segmenter.mask_generator = True # Just needs to be non-None for the check
    return segmenter

@pytest.fixture
def real_segmenter_instance():
    """
    Provides an instance of ImageSegmenter with an attempt to load a real model.
    NOTE: User needs to replace 'YOUR_MODEL_CHECKPOINT_PATH.pth' with the actual path.
    This fixture also assumes ImageSegmenter._load_model() is functional.
    """
    checkpoint_path = "/home/yibei/sam2/checkpoints/sam2.1_hiera_large.pt" # <--- USER: REPLACE THIS
    model_type = "hiera_l" # Or your chosen SAM2 model type
    device = "cuda" # Or "cpu"

    if checkpoint_path == "YOUR_MODEL_CHECKPOINT_PATH.pth" or not os.path.exists(checkpoint_path):
        pytest.skip("Real model checkpoint path not configured or not found. Skipping video segmentation test.")
    
    try:
        segmenter = ImageSegmenter(model_type=model_type, checkpoint_path=checkpoint_path, device=device)
        if segmenter.predictor is None or segmenter.mask_generator is None: # Check if model loading likely failed
            pytest.skip("ImageSegmenter real model not loaded (predictor or generator is None). Skipping video test.")
        return segmenter
    except Exception as e:
        pytest.skip(f"Failed to initialize ImageSegmenter with real model: {e}. Skipping video test.")

def test_segment_human_and_background(segmenter_instance, dummy_image):
    """Tests the segment_human method and background mask derivation."""
    segmenter = segmenter_instance
    image = dummy_image
    h, w, _ = image.shape
    
    # Define a dummy bounding box within the image dimensions
    face_bbox = [w // 4, h // 4, w * 3 // 4, h * 3 // 4]
    
    human_mask, background_mask = segmenter.segment_human(image, face_bbox=face_bbox)
    
    # Check return types and shapes (based on placeholder implementation)
    assert human_mask is not None
    assert background_mask is not None
    assert isinstance(human_mask, np.ndarray)
    assert isinstance(background_mask, np.ndarray)
    assert human_mask.shape == (h, w)
    assert background_mask.shape == (h, w)
    assert human_mask.dtype == bool
    assert background_mask.dtype == bool
    
    # Check relationship between human and background masks
    assert np.all(background_mask == ~human_mask)
    
    # Check placeholder logic specifically (human mask should cover the bbox area)
    x1, y1, x2, y2 = [int(c) for c in face_bbox]
    assert np.sum(human_mask) == (x2 - x1) * (y2 - y1) # Check number of True pixels
    assert np.all(human_mask[y1:y2, x1:x2]) # Check if the bbox area is True
    assert np.sum(human_mask) + np.sum(background_mask) == h * w # Check total coverage

def test_segment_all(segmenter_instance, dummy_image):
    """Tests the segment_all method."""
    segmenter = segmenter_instance
    image = dummy_image
    h, w, _ = image.shape

    all_masks = segmenter.segment_all(image)

    # Check return type (based on placeholder implementation)
    assert all_masks is not None
    assert isinstance(all_masks, list)
    
    # Placeholder currently returns an empty list
    assert len(all_masks) == 0
    
    # If the placeholder were updated to return dummy masks, 
    # we would add checks here like:
    # if all_masks:
    #     assert isinstance(all_masks[0], np.ndarray)
    #     assert all_masks[0].shape == (h, w)
    #     assert all_masks[0].dtype == bool 

def test_segment_human_without_bbox(segmenter_instance, dummy_image):
    """Tests the segment_human method when no face_bbox is provided."""
    segmenter = segmenter_instance
    image = dummy_image
    
    # Call segment_human with face_bbox=None
    human_mask, background_mask = segmenter.segment_human(image, face_bbox=None)
    
    # For the placeholder, if no bbox is given, it's expected to return None for masks.
    # This behavior might change if the actual model can perform unprompted human segmentation.
    assert human_mask is None
    assert background_mask is None 

# New test for video segmentation
def test_segment_human_on_video(real_segmenter_instance):
    """
    Tests human segmentation on a video, saving the output.
    Prompts for human segmentation are NOT used (face_bbox=None).
    NOTE: User needs to replace 'YOUR_TEST_VIDEO.mp4'.
    """
    segmenter = real_segmenter_instance # Uses the new fixture with a real model
    
    video_path = "/home/yibei/facetracker/tests/friends_s02e09a_20s_slice.mp4" # <--- USER: REPLACE THIS with your 20s video
    output_dir = "tests/output/segmentation_video_test"
    output_video_filename = "human_segmentation_output.mp4"

    if video_path == "/home/yibei/facetracker/tests/friends_s02e09a_20s_slice.mp4" or not os.path.exists(video_path):
        pytest.skip("Test video path not configured or video not found. Skipping video segmentation test.")

    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pytest.fail(f"Could not open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter object
    # Using 'mp4v' for .mp4 output, adjust if needed for other formats/OS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

    print(f"Processing video {video_path} for human segmentation (no bbox prompt)...")
    frame_count = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Perform human segmentation without a bounding box prompt
        human_mask_bool, _ = segmenter.segment_human(frame_rgb, face_bbox=None)
        
        composite_frame = np.zeros((frame_height, frame_width * 2, 3), dtype=np.uint8)
        composite_frame[:, :frame_width] = frame_bgr # Original frame on the left

        if human_mask_bool is not None:
            # Ensure mask is boolean, then convert to 0/255 for visualization
            if human_mask_bool.dtype != bool: # Should be bool from segmenter spec
                human_mask_bool = human_mask_bool > 0.5 # Example threshold if it's float
            
            human_mask_viz = (human_mask_bool.astype(np.uint8) * 255)
            human_mask_viz_colored = cv2.cvtColor(human_mask_viz, cv2.COLOR_GRAY2BGR)
            
            # Create segmented person view: original * mask
            # Ensure mask is broadcastable: (H, W, 1)
            segmented_person = cv2.bitwise_and(frame_bgr, frame_bgr, mask=human_mask_bool.astype(np.uint8))
            
            # Place segmented person on the right part of composite frame
            composite_frame[:, frame_width:] = segmented_person
            
            # Optional: Draw text if segmentation was successful
            cv2.putText(composite_frame, "Human Segmented (No Prompt)", (frame_width + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # If no human mask, show original frame on the right too, or a placeholder
            composite_frame[:, frame_width:] = frame_bgr 
            cv2.putText(composite_frame, "No Human Mask (No Prompt)", (frame_width + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        out_video.write(composite_frame)
        frame_count +=1
        if frame_count % int(fps) == 0: # Log progress every second
            print(f"Processed {frame_count} frames...")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output video saved to: {output_video_path}")
    assert os.path.exists(output_video_path), "Output video file was not created." 