"""Video and webcam processing pipeline."""

import os
import cv2
from collections import defaultdict

from vehicle_tracker import VehicleTracker
from plate_detector import PlateDetector
from visualizer import Visualizer
from config import (
    DEFAULT_MODEL_PATH, DEFAULT_PLATE_MODEL_PATH,
    CROPPED_FOLDER, MAX_PLATE_DETECTIONS_PER_VEHICLE
)


class VideoProcessor:
    """Processes video/webcam with vehicle tracking and plate detection."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence=0.5,
                 input_size=(384, 384), use_gpu=True, track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, plate_model=DEFAULT_PLATE_MODEL_PATH,
                 plate_confidence=0.5, enable_plates=True, cropped_folder=CROPPED_FOLDER,
                 max_plate_detections=MAX_PLATE_DETECTIONS_PER_VEHICLE):
        
        self.tracker = VehicleTracker(
            model_path, confidence, input_size, use_gpu,
            track_thresh, track_buffer, match_thresh
        )
        self.visualizer = Visualizer()
        self.enable_plates = enable_plates
        self.cropped_folder = cropped_folder
        self.max_plate_detections = max_plate_detections
        self.plate_counts = defaultdict(int)
        
        self.plate_detector = None
        if enable_plates:
            try:
                self.plate_detector = PlateDetector(
                    plate_model, plate_confidence, input_size, use_gpu
                )
                os.makedirs(cropped_folder, exist_ok=True)
            except Exception as e:
                print(f"Plate detector unavailable: {e}")
                self.enable_plates = False

    def _detect_plates(self, frame, detections, frame_num):
        """Detect plates for vehicles in bottom half."""
        if not self.enable_plates or detections.tracker_id is None:
            return []
        
        h = frame.shape[0]
        results = []
        
        for tid, box in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            
            # Skip if not in bottom half or already at max detections
            if (y1 + y2) / 2 < h / 2:
                continue
            if self.plate_counts[tid] >= self.max_plate_detections:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            plates = self.plate_detector.detect(crop)
            if plates:
                self.plate_counts[tid] += 1
                count = self.plate_counts[tid]
                
                for p in plates:
                    px1, py1, px2, py2 = p['box']
                    plate_crop = crop[py1:py2, px1:px2]
                    
                    if plate_crop.size > 0:
                        filename = f"{tid}_{count}_f{frame_num}.jpg"
                        path = os.path.join(self.cropped_folder, filename)
                        cv2.imwrite(path, plate_crop)
                        results.append({'tracker_id': tid, 'vehicle_box': [x1, y1, x2, y2]})
                        print(f"Plate #{tid} ({count}/{self.max_plate_detections}): {filename}")
        
        return results

    def process_video(self, video_path, output_path=None, show=False, show_trace=True):
        """Process video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.tracker.reset()
        self.tracker.set_frame_rate(fps)
        self.plate_counts.clear()
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_num = 0
        all_tracks = set()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                detections = self.tracker.track(frame)
                
                if detections.tracker_id is not None:
                    all_tracks.update(detections.tracker_id.tolist())
                
                plate_results = self._detect_plates(frame, detections, frame_num)
                
                # Annotate
                annotated = self.visualizer.annotate(frame, detections, show_trace)
                self.visualizer.draw_zone_line(annotated, h // 2, "Plate Detection Zone")
                
                for pr in plate_results:
                    self.visualizer.highlight_box(annotated, pr['vehicle_box'])
                
                plates_saved = sum(self.plate_counts.values())
                self.visualizer.draw_stats(
                    annotated,
                    f"Frame: {frame_num}/{total} | Vehicles: {len(detections)} | Plates: {plates_saved}"
                )
                
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Vehicle Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if frame_num % 30 == 0:
                    print(f"Frame {frame_num}/{total} | Vehicles: {len(detections)} | Plates: {plates_saved}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        self._print_summary(frame_num, all_tracks, output_path)

    def process_webcam(self, camera_id=0, show_trace=True):
        """Process live webcam feed."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        self.tracker.reset()
        all_tracks = set()
        
        print("Press 'q' to quit, 'r' to reset")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.tracker.track(frame)
                
                if detections.tracker_id is not None:
                    all_tracks.update(detections.tracker_id.tolist())
                
                annotated = self.visualizer.annotate(frame, detections, show_trace)
                self.visualizer.draw_stats(
                    annotated, f"Active: {len(detections)} | Total: {len(all_tracks)}"
                )
                
                cv2.imshow("Vehicle Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.tracker.reset()
                    all_tracks.clear()
                    print("Reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"Total vehicles tracked: {len(all_tracks)}")

    def _print_summary(self, frames, tracks, output_path):
        """Print processing summary."""
        print(f"\n{'='*40}")
        print(f"Frames processed: {frames}")
        print(f"Unique vehicles: {len(tracks)}")
        print(f"Plates saved: {sum(self.plate_counts.values())}")
        print(f"Vehicles with plates: {len(self.plate_counts)}")
        if output_path:
            print(f"Output: {output_path}")
        if self.enable_plates:
            print(f"Plates folder: {self.cropped_folder}")
        print('='*40)
