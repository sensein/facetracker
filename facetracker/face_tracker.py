import cv2
import numpy as np
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import pandas as pd

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.predict_num = 0
        self.additional_attributes = []

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if len(bbox) > 0:
            self.kf.update(self.convert_bbox_to_z(bbox))
            self.predict_num = 0
        else:
            self.predict_num += 1

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.get_state()

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, additional_attributes=None, predict_num=5):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
          img_size - [height, width] of the image
          additional_attributes - list of additional attributes for each detection
          predict_num - maximum number of frames to predict when unmatched
        Returns a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        if len(dets) > 0:
            matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(self, detections, trackers, self.iou_threshold)

            # Update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    if additional_attributes:
                        trk.additional_attributes.append(additional_attributes[d[0]])

            # Create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                if additional_attributes:
                    trk.additional_attributes.append(additional_attributes[i])
                self.trackers.append(trk)
        
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if len(dets) == 0:
                trk.update([])
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # Remove dead tracklet
            if (trk.time_since_update >= self.max_age or 
                trk.predict_num >= predict_num or 
                d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]):
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold = 0.3):
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

        for d,det in enumerate(detections):
            for t,trk in enumerate(trackers):
                iou_matrix[d,t] = self.iou(det,trk)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices)
        matched_indices = np.transpose(matched_indices)

        unmatched_detections = []
        for d,det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t,trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0],m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                  + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return(o)

class FaceTracker:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.5):
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.next_global_id = 0
        self.face_data: Dict[int, Dict[str, Any]] = {}
        self.sort_id_to_global_id: Dict[int, int] = {}
        self.global_id_to_confidence: Dict[int, float] = {}

    def track_faces(self, frame: int, face_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"Input face_data: {face_data}")
        
        if not face_data:
            tracked_faces = self.sort_tracker.update()
        else:
            detections = np.array([face['bbox'] + [face['confidence']] for face in face_data])
            print(f"Detections passed to SORT: {detections}")
            tracked_faces = self.sort_tracker.update(detections)
        
        print(f"Output from SORT tracker: {tracked_faces}")
        
        result = []
        
        for i, face in enumerate(tracked_faces):
            sort_id = int(face[4])
            if sort_id not in self.sort_id_to_global_id:
                global_id = self.next_global_id
                self.next_global_id += 1
                self.sort_id_to_global_id[sort_id] = global_id
                self.global_id_to_confidence[global_id] = face_data[i]['confidence'] if i < len(face_data) else 1.0
            else:
                global_id = self.sort_id_to_global_id[sort_id]
            
            face_info = {
                "id": global_id,
                "frame": frame,
                "bbox": face[:4].tolist(),
                "confidence": self.global_id_to_confidence[global_id]
            }
            print(f"Created face_info: {face_info}")
            self.face_data[global_id] = face_info
            result.append(face_info)
        
        print(f"Final result: {result}")
        return result

    def get_face_data(self) -> Dict[int, Dict[str, Any]]:
        return self.face_data
    
class FrameSelector:
    def __init__(
        self,
        video_file: str,
        top_n: int = 3,
        output_dir: Optional[str] = None,
        save_images: bool = True,
    ):
        self.video_file = video_file
        self.top_n = top_n
        self.output_dir = output_dir
        self.save_images = save_images

        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        return cv2.mean(image)[0]

    @staticmethod
    def calculate_blurriness(image: np.ndarray) -> float:
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def save_cropped_face(
        self, face_image: np.ndarray, global_face_id: int, frame_idx: int
    ) -> Optional[str]:
        if self.output_dir and self.save_images:
            save_filename = f"face_{global_face_id}_frame_{frame_idx}.jpg"
            save_path = os.path.join(self.output_dir, save_filename)
            cv2.imwrite(save_path, face_image)
            return save_filename
        return None

    def select_top_frames_per_face(
        self, tracked_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        cap = cv2.VideoCapture(self.video_file)
        face_data: Dict[int, List[Dict[str, Any]]] = {}

        total_faces = sum(len(scene_faces) for scene_faces in tracked_data.values())

        with tqdm(total=total_faces, desc="Processing faces") as pbar:
            for scene_id, scene_faces in tracked_data.items():
                for face_entry in scene_faces:
                    global_face_id = face_entry["id"]
                    frame_idx = face_entry["frame"]
                    face_coords = face_entry["face"]
                    confidence = face_entry["conf"]

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx}. Skipping.")
                        continue

                    x1, y1, x2, y2 = map(int, face_coords)
                    height, width = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    face_image = frame[y1:y2, x1:x2]
                    if face_image.size == 0:
                        print(f"Warning: Empty face image in frame {frame_idx}. Skipping.")
                        continue

                    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_size = (x2 - x1) * (y2 - y1)
                    brightness = self.calculate_brightness(gray_face)
                    blurriness = self.calculate_blurriness(gray_face)

                    normalized_face_size = face_size / (width * height)
                    normalized_brightness = brightness / 255.0
                    normalized_blurriness = blurriness / (blurriness + 1e-6)

                    score = (
                        confidence
                        + 0.5 * normalized_face_size
                        + 0.3 * normalized_brightness
                        - 0.2 * normalized_blurriness
                    )

                    relative_path = self.save_cropped_face(face_image, global_face_id, frame_idx)

                    if global_face_id not in face_data:
                        face_data[global_face_id] = []

                    face_data[global_face_id].append({
                        "frame_idx": frame_idx,
                        "total_score": score,
                        "face_coord": face_coords,
                        "image_path": relative_path,
                        "scene_id": scene_id
                    })

                    pbar.update(1)

        cap.release()

        selected_frames: Dict[str, List[Dict[str, Any]]] = {}
        for global_face_id, frames in face_data.items():
            top_frames = sorted(frames, key=lambda x: x["total_score"], reverse=True)[:self.top_n]
            
            for frame in top_frames:
                scene_id = frame["scene_id"]
                if scene_id not in selected_frames:
                    selected_frames[scene_id] = []

                selected_frames[scene_id].append({
                    "global_face_id": f"global_face_{global_face_id}",
                    "top_frames": [{
                        "frame_idx": f["frame_idx"],
                        "total_score": f["total_score"],
                        "face_coord": f["face_coord"],
                        "image_path": f["image_path"]
                    } for f in top_frames if f["scene_id"] == scene_id]
                })

        return selected_frames
