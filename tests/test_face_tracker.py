import unittest
import numpy as np
import pandas as pd
from facetracker.face_tracker import FaceTracker
from typing import Dict, List, Any

class TestFaceTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = FaceTracker(max_age=1, min_hits=3, iou_threshold=0.1)

    def test_track_faces_single_face(self):
        face_data = [{'bbox': [100, 100, 200, 200], 'confidence': 0.9}]
        print(f"\nInput to track_faces: {face_data}")
        result = self.tracker.track_faces(1, face_data)
        print(f"Output from track_faces: {result}")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 0)
        self.assertEqual(result[0]['frame'], 1)
        self.assertEqual(result[0]['bbox'], [100, 100, 200, 200])
        self.assertAlmostEqual(result[0]['confidence'], 0.9, places=6)

    def test_get_face_data(self):
        face_data = [{'bbox': [100, 100, 200, 200], 'confidence': 0.9}]
        print(f"\nInput to track_faces: {face_data}")
        self.tracker.track_faces(1, face_data)
        
        face_data = self.tracker.get_face_data()
        print(f"Output from get_face_data: {face_data}")
        self.assertEqual(len(face_data), 1)
        self.assertIn(0, face_data)
        self.assertEqual(face_data[0]['frame'], 1)
        self.assertEqual(face_data[0]['bbox'], [100, 100, 200, 200])
        self.assertAlmostEqual(face_data[0]['confidence'], 0.9, places=6)

    def test_complex_tracking_scenario(self):
        # Generate test data
        face_data = [
            # Frame 1: Face 1 appears
            [{'bbox': [100, 100, 200, 200], 'confidence': 0.9}],
            # Frame 2: Face 1 moves slightly, Face 2 appears
            [
                {'bbox': [102, 98, 202, 198], 'confidence': 0.92},
                {'bbox': [300, 300, 400, 400], 'confidence': 0.85}
            ],
            # Frame 3: Face 1 disappears, Face 2 moves, Face 3 appears
            [
                {'bbox': [305, 295, 405, 395], 'confidence': 0.87},
                {'bbox': [500, 500, 600, 600], 'confidence': 0.95}
            ],
            # Frame 4: No faces
            [],
            # Frame 5: Face 1 reappears, Face 2 and 3 are present
            [
                {'bbox': [98, 102, 198, 202], 'confidence': 0.91},
                {'bbox': [310, 290, 410, 390], 'confidence': 0.88},
                {'bbox': [495, 505, 595, 605], 'confidence': 0.96}
            ],
            # Frame 6: Face 1 and 3 present, Face 2 disappears
            [
                {'bbox': [97, 103, 197, 203], 'confidence': 0.93},
                {'bbox': [490, 510, 590, 610], 'confidence': 0.97}
            ]
        ]

        all_results = []
        for frame, faces in enumerate(face_data, start=1):
            print(f"\nProcessing Frame {frame}")
            result = self.tracker.track_faces(frame, faces)
            all_results.extend(result)
            print(f"Frame {frame} Results: {result}")

        # Assertions
        self.assertEqual(len(all_results), 11)  # Total number of face detections across all frames

        # Check if we have 3 unique face IDs
        unique_ids = set(face['id'] for face in all_results)
        self.assertEqual(len(unique_ids), 3)

        # Check if faces appear in the correct frames
        frame_face_counts = [len(self.tracker.track_faces(i+1, faces)) for i, faces in enumerate(face_data)]
        self.assertEqual(frame_face_counts, [1, 2, 2, 0, 3, 2])

        # Check if bounding boxes are oscillating
        face_1_bboxes = [face['bbox'] for face in all_results if face['id'] == 0]
        self.assertNotEqual(face_1_bboxes[0], face_1_bboxes[-1])

        # Check if confidences are preserved
        self.assertAlmostEqual(all_results[0]['confidence'], 0.9, places=2)
        self.assertAlmostEqual(all_results[-1]['confidence'], 0.97, places=2)

        print("\nAll tracking results:")
        for face in all_results:
            print(f"Frame {face['frame']}, ID {face['id']}: {face['bbox']}, Confidence: {face['confidence']}")

if __name__ == '__main__':
    unittest.main()