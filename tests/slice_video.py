import cv2
import os

def slice_video(
    input_video_path: str,
    output_video_path: str,
    start_time_seconds: int,
    end_time_seconds: int,
):
    """
    Slices a video from start_time_seconds to end_time_seconds and saves it.

    Args:
        input_video_path: Path to the long input video.
        output_video_path: Path where the sliced video will be saved.
        start_time_seconds: Start time for the slice in seconds.
        end_time_seconds: End time for the slice in seconds.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get FPS from video. Cannot slice accurately.")
        cap.release()
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frames / fps

    print(f"Input video FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {video_duration_seconds:.2f}s")

    if start_time_seconds < 0:
        print("Warning: start_time_seconds is negative, setting to 0.")
        start_time_seconds = 0
    if end_time_seconds > video_duration_seconds:
        print(f"Warning: end_time_seconds ({end_time_seconds}s) exceeds video duration ({video_duration_seconds:.2f}s). "
              f"Will slice until the end of the video.")
        end_time_seconds = video_duration_seconds
    if start_time_seconds >= end_time_seconds:
        print("Error: start_time_seconds must be less than end_time_seconds.")
        cap.release()
        return

    start_frame = int(start_time_seconds * fps)
    end_frame = int(end_time_seconds * fps)

    print(f"Slicing from {start_time_seconds}s (frame {start_frame}) to {end_time_seconds}s (frame {end_frame}).")

    # Get video properties for the writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    # Using 'mp4v' for .mp4, or 'XVID' for .avi often works.
    # You might need to experiment with codecs based on your system (e.g., 'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out_writer.isOpened():
        print(f"Error: Could not open VideoWriter for path {output_video_path}. Check codec and permissions.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_num = start_frame

    print(f"Starting to write frames to {output_video_path}...")
    frames_written = 0
    while current_frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {current_frame_num}. Stopping early.")
            break
        
        out_writer.write(frame)
        frames_written += 1
        current_frame_num += 1
        
        if current_frame_num % int(fps * 10) == 0: # Print progress every 10 seconds of video
            print(f"Processed up to frame {current_frame_num}/{end_frame}...")

    print(f"Finished writing. {frames_written} frames written to {output_video_path}.")

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows() # Just in case any windows were opened by OpenCV

# --- Example Usage ---
if __name__ == "__main__":
    long_video_path = "/orcd/scratch/bcs/001/yibei/friends/data/mkv2mp4/friends_s02e09a.mp4" 
    output_sliced_video_path = "/home/yibei/facetracker/tests/friends_s02e09a_20s_slice.mp4"
    
    start_seconds = 60  # Start at 1 minute (for example)
    duration_seconds = 20 # Slice for 1 minute
    end_seconds = start_seconds + duration_seconds

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_sliced_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    slice_video(long_video_path, output_sliced_video_path, start_seconds, end_seconds)
    print(f"Video sliced and saved to {output_sliced_video_path}")
