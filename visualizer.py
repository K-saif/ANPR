"""Visualization and annotation utilities."""

import cv2
import supervision as sv
from vehicle_tracker import VehicleTracker


class Visualizer:
    """Handles frame annotation and visualization."""
    
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=50, position=sv.Position.BOTTOM_CENTER
        )

    def annotate(self, frame, detections, show_trace=True):
        """Draw boxes, labels, and traces on frame."""
        if len(detections) == 0:
            return frame.copy()
        
        labels = self._build_labels(detections)
        annotated = frame.copy()
        
        if show_trace and detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(annotated, detections)
        
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        return annotated

    def _build_labels(self, detections):
        """Generate labels for detections."""
        labels = []
        has_ids = detections.tracker_id is not None
        
        for i, (cid, conf) in enumerate(zip(detections.class_id, detections.confidence)):
            name = VehicleTracker.get_class_name(cid)
            if has_ids:
                labels.append(f"#{detections.tracker_id[i]} {name} {conf:.2f}")
            else:
                labels.append(f"{name} {conf:.2f}")
        return labels

    @staticmethod
    def draw_zone_line(frame, y_pos, label="Detection Zone"):
        """Draw horizontal zone line."""
        h, w = frame.shape[:2]
        cv2.line(frame, (0, y_pos), (w, y_pos), (0, 255, 255), 2)
        cv2.putText(frame, label, (10, y_pos + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    @staticmethod
    def draw_stats(frame, text):
        """Draw stats overlay."""
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    @staticmethod
    def highlight_box(frame, box, color=(0, 0, 255), thickness=3):
        """Highlight a bounding box."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
