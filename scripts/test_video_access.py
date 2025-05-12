import cv2
import sys

video_path = "/input_videos/friends_s01e01a.mkv"
print(f"Attempting to open video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)
else:
    print(f"Successfully opened video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"  FPS: {fps}")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  Frame Count: {frame_count}")
    cap.release()
    print("Video capture released.")
sys.exit(0) 