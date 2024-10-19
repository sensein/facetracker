import cv2
import os
from facenet_pytorch import MTCNN
import torch
import json
import numpy as np
from tqdm import tqdm

class FaceDetector:
    def __init__(self, video_path, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu', min_confidence=0.8):
        self.video_path = video_path
        self.output_dir = output_dir
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device, factor=0.6)  # Use MTCNN for face detection with scaling factor
        self.min_confidence = min_confidence
    
    def detect_faces_in_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():  # Check if the video file was opened successfully
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        annotated_video_path = os.path.join(self.output_dir, f"{os.path.basename(self.video_path).split('.')[0]}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, int(fps), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        face_detections = []

        for frame_idx in tqdm(range(total_frames), desc="Processing video frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            faces, annotated_frame = self._detect_and_annotate_frame(frame)
            out.write(annotated_frame)

            face_detections.append({
                "frame_index": frame_idx,
                "detections": faces
            })

        cap.release()
        out.release()

        return face_detections

    def _detect_and_annotate_frame(self, frame):
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                if prob >= self.min_confidence and self._is_valid_box(box):
                    faces.append({
                        "box": box.tolist(), 
                        "confidence": prob, 
                        "landmarks": landmark.tolist()
                    })
        
        annotated_frame = self._annotate_frame(frame, faces)
        return faces, annotated_frame

    def _annotate_frame(self, frame, faces, rect_color=(0, 255, 0), circle_color=(0, 0, 255), text_color=(255, 255, 255)):
        for face in faces:
            box = face['box']
            landmarks = face['landmarks']
            confidence = face['confidence']
            
            # Draw the bounding box
            cv2.rectangle(frame, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          rect_color, 2)
            
            # Draw the landmarks
            for point in landmarks:
                cv2.circle(frame, 
                           (int(point[0]), int(point[1])), 
                           2, 
                           circle_color, -1)
            
            # Display the confidence level
            cv2.putText(frame, 
                        f'{confidence:.2f}', 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        text_color, 2)
        return frame

    def _is_valid_box(self, box, min_size=20, max_aspect_ratio=1.5):
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = max(width / height, height / width)
        return width > min_size and height > min_size and aspect_ratio <= max_aspect_ratio

    def save_results(self, output_file, face_detections):
        with open(output_file, 'w') as f:
            json.dump(face_detections, f, indent=4)