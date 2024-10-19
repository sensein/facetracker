import cv2
from scenedetect import SceneManager, VideoCaptureAdapter
from scenedetect.detectors import ContentDetector, AdaptiveDetector, HashDetector

class SceneDetector:
    def __init__(self, video_path, detector_type='adaptive', min_scene_len=15):
        self.video_path = video_path
        self.detector_type = detector_type.lower()
        self.min_scene_len = min_scene_len
        self.cap = None
        self.scene_manager = None
        self.shots = None

    def initialize_scene_manager(self):
        # Initialize SceneManager with the chosen detector
        self.scene_manager = SceneManager()
        if self.detector_type == 'content':
            self.scene_manager.add_detector(ContentDetector(min_scene_len=self.min_scene_len))
        elif self.detector_type == 'hash':
            self.scene_manager.add_detector(HashDetector(min_scene_len=self.min_scene_len))
        else:  # Default to AdaptiveDetector
            self.scene_manager.add_detector(AdaptiveDetector(min_scene_len=self.min_scene_len))
    
    def detect_scenes(self):
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Enable GPU acceleration in OpenCV (if available)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("GPU acceleration enabled.")
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        adapter = VideoCaptureAdapter(self.cap)
        self.scene_manager.detect_scenes(frame_source=adapter)
        self.shots = self.get_shot_list()
        self.cap.release()

    def get_shot_list(self):
        # Calculate frame rate and convert frame numbers to seconds
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        scene_list = self.scene_manager.get_scene_list()
        shots = [(scene[0].get_frames(), scene[1].get_frames(),
                  scene[0].get_frames() / fps, scene[1].get_frames() / fps) for scene in scene_list]
        return shots
    
    def save_shots(self, output_file):
        # Save shot boundaries with both frame numbers and time in seconds
        with open(output_file, 'w') as f:
            f.write("Start Frame,End Frame,Start Time (s),End Time (s)\n")
            for start_frame, end_frame, start_time, end_time in self.shots:
                f.write(f"{start_frame},{end_frame},{start_time:.2f},{end_time:.2f}\n")
    
    def load_shots(self, input_file):
        with open(input_file, 'r') as f:
            self.shots = [tuple(map(float, line.strip().split(','))) for line in f.readlines()[1:]]
    
    def print_shots(self):
        # Print detected shots with both frame numbers and seconds
        for i, (start_frame, end_frame, start_time, end_time) in enumerate(self.shots):
            print(f"Shot {i + 1}: Start Frame = {start_frame}, End Frame = {end_frame}, Start Time = {start_time:.2f}s, End Time = {end_time:.2f}s")