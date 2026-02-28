"""Vehicle detection and tracking using ONNX + ByteTrack."""

import numpy as np
import supervision as sv
from collections import defaultdict

from base_detector import BaseDetector
from config import DEFAULT_MODEL_PATH, VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES


class VehicleTracker(BaseDetector):
    """Detects and tracks vehicles using ByteTrack."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=True, track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, frame_rate=30):
        super().__init__(model_path, confidence_threshold, input_size, use_gpu)
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        self.track_history = defaultdict(list)
        self.max_history = 50

    def detect(self, image):
        """Detect vehicles, returns supervision.Detections."""
        input_tensor, (orig_w, orig_h) = self.preprocess(image)
        outputs = self.run_inference(input_tensor)
        
        boxes, logits = outputs[0][0], outputs[1][0]
        probs = 1 / (1 + np.exp(-logits))
        scores = np.max(probs, axis=1)
        labels = np.argmax(probs, axis=1)
        
        # Filter by confidence and vehicle classes
        mask = (scores >= self.confidence_threshold) & np.isin(labels, VEHICLE_CLASS_IDS)
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
        
        if len(boxes) == 0:
            return sv.Detections.empty()
        
        # Convert [cx,cy,w,h] normalized to [x1,y1,x2,y2] absolute
        cx, cy, w, h = boxes.T
        xyxy = np.stack([
            (cx - w/2) * orig_w, (cy - h/2) * orig_h,
            (cx + w/2) * orig_w, (cy + h/2) * orig_h
        ], axis=1)
        
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int)
        )

    def track(self, image):
        """Detect and track vehicles."""
        detections = self.detect(image)
        if len(detections) == 0:
            return detections
        
        tracked = self.tracker.update_with_detections(detections)
        
        # Update trajectory history
        if tracked.tracker_id is not None:
            for tid, box in zip(tracked.tracker_id, tracked.xyxy):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                history = self.track_history[tid]
                history.append(center)
                if len(history) > self.max_history:
                    history.pop(0)
        
        return tracked

    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.track_history.clear()

    def set_frame_rate(self, fps):
        """Reinitialize tracker with new frame rate."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=fps
        )

    @staticmethod
    def get_class_name(class_id):
        return VEHICLE_CLASS_NAMES.get(class_id, 'vehicle')
