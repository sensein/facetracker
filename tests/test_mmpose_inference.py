import pytest
import os
from torch.serialization import add_safe_globals
import numpy.core.multiarray
import numpy as np

# Add numpy's _reconstruct, ndarray, dtype, all dtypes, and scalar to safe globals
add_safe_globals([numpy.core.multiarray._reconstruct, np.ndarray, np.dtype, numpy.core.multiarray.scalar])
# Add all numpy dtypes
for dtype in [getattr(np, name) for name in dir(np) if isinstance(getattr(np, name), type) and issubclass(getattr(np, name), np.generic)]:
    add_safe_globals([dtype])
if hasattr(np, 'dtypes'):
    for dtype in np.dtypes.__all__:
        add_safe_globals([getattr(np.dtypes, dtype)])
# Add mmengine.logging.history_buffer.HistoryBuffer
try:
    from mmengine.logging.history_buffer import HistoryBuffer
    add_safe_globals([HistoryBuffer])
except ImportError:
    pass

# Attempt to import mmpose modules and skip test if not available
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules
    MMPROSE_INSTALLED = True
except ImportError:
    MMPROSE_INSTALLED = False

# Define paths to model files and demo image
# Assumes these files are placed in the project root directory (e.g., /home/yibei/facetracker/)
CONFIG_FILE_NAME = '/home/yibei/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
CHECKPOINT_FILE_NAME = '/home/yibei/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
DEMO_IMAGE_NAME = '/home/yibei/demo.jpg'

# Construct full paths relative to the current workspace might be more robust
# For now, assuming direct execution from project root or correct pytest setup for paths.

@pytest.mark.skipif(not MMPROSE_INSTALLED, reason="mmpose is not successfully installed in the environment.")
def test_mmpose_hrnet_inference():
    """
    Tests basic top-down inference with the downloaded HRNet MMPose model.

    Prerequisites:
    1. MMPose and its dependencies must be installed (e.g., via 'poetry install').
    2. The model config file (CONFIG_FILE_NAME) must be in the project root.
    3. The model checkpoint file (CHECKPOINT_FILE_NAME) must be in the project root.
    4. A demo image (DEMO_IMAGE_NAME) with at least one person must be in the project root.
    """
    # Ensure model and image files exist where expected
    assert os.path.exists(CONFIG_FILE_NAME), \
        f"Config file '{CONFIG_FILE_NAME}' not found in project root. Please place it there."
    assert os.path.exists(CHECKPOINT_FILE_NAME), \
        f"Checkpoint file '{CHECKPOINT_FILE_NAME}' not found in project root. Please place it there."
    assert os.path.exists(DEMO_IMAGE_NAME), \
        f"Demo image '{DEMO_IMAGE_NAME}' not found in project root. Please create or place it there."

    # Register all MMPose modules
    register_all_modules()

    # Initialize the model
    # Using 'cpu' for device to make the test more portable and avoid GPU dependencies.
    # User can change to 'cuda:0' if a GPU is available and testing GPU inference.
    model = init_model(CONFIG_FILE_NAME, CHECKPOINT_FILE_NAME, device='cpu')
    assert model is not None, "Failed to initialize the MMPose model."

    # Perform inference
    # 'inference_topdown' expects the image path and optional bounding boxes.
    # For simple testing, we assume the model handles person detection or the image is cropped.
    results = inference_topdown(model, DEMO_IMAGE_NAME)

    # Basic assertions on the results
    assert results is not None, "Inference results should not be None."
    assert isinstance(results, list), "Inference results should be a list."
    
    if len(results) > 0:
        # Each element in the results list is now a PoseDataSample object
        first_person_result = results[0]
        # Check that pred_instances exists and has keypoints
        assert hasattr(first_person_result, 'pred_instances'), "Result should have pred_instances"
        assert hasattr(first_person_result.pred_instances, 'keypoints'), "pred_instances should have keypoints"
        keypoints = first_person_result.pred_instances.keypoints
        assert isinstance(keypoints, np.ndarray), "keypoints should be a NumPy array"
        print(f"Keypoints shape: {keypoints.shape}")
        
        # The keypoints array has shape (N, K, 2) where:
        # N is the number of instances (people)
        # K is the number of keypoints per person
        # 2 is for x,y coordinates
        assert len(keypoints.shape) == 3, "Keypoints should be a 3D array (N, K, 2)"
        assert keypoints.shape[2] == 2, "Each keypoint should have x,y coordinates"
        
        # Check that keypoints are within image bounds
        # Note: The model outputs coordinates in the original image space
        img_shape = first_person_result.metainfo['img_shape']
        for person_kps in keypoints:
            for kp in person_kps:
                assert 0 <= kp[0] <= img_shape[1], "Keypoint x-coordinate is out of bounds"
                assert 0 <= kp[1] <= img_shape[0], "Keypoint y-coordinate is out of bounds"
    else:
        print("No persons detected in the demo image by the model.")
