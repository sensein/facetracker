# In test_tf.py
import os
# Set environment variables for verbose TensorFlow logging BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# Try a more specific module for VLOG, focusing on GPU and library loading
os.environ['TF_CPP_VMODULE'] = 'gpu_init=2,select_platform=2,dynamic_library_loader=2,cuda_driver=2,gpu_config=2,plugin_fft=2,plugin_dnn=2,plugin_blas=2'
# Or even more general if the above doesn't yield much:
# os.environ['TF_CPP_VMODULE'] = '<em>=2' # This will be VERY verbose

print(f"Python's initial LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

import tensorflow as tf

print("TensorFlow imported successfully.")
print(f"TF Version: {tf.__version__}")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Available GPUs:", physical_devices)
    for device in physical_devices:
        details = tf.config.experimental.get_device_details(device)
        print(f"Details for {device.name}: {details.get('device_name', 'N/A')}")
else:
    print("No GPU detected by TensorFlow.")

print(f"LD_LIBRARY_PATH after TF import: {os.environ.get('LD_LIBRARY_PATH')}")

# You can also add a small TF operation to see if it defaults to CPU or errors out further
# For example:
try:
    with tf.device('/gpu:0'): # Or try without explicitly placing on GPU first
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    print("Matrix multiplication on GPU successful:", c)
except RuntimeError as e:
    print("RuntimeError during TF operation (likely GPU issue):", e)
except Exception as e:
    print("Exception during TF operation:", e)
