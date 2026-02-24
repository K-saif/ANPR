"""
ONNX Vehicle Detection + Tracking Script
Uses RF-DETR ONNX model with ByteTrack for vehicle tracking
"""

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import supervision as sv
from collections import defaultdict

# Vehicle class IDs from COCO dataset
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Default ONNX model path
DEFAULT_MODEL_PATH = "/home/medprime/Music/ANPR/output/inference_model.onnx"


class VehicleTracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=True,
                 track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        """
        Initialize the ONNX vehicle detector with tracking.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            input_size: Model input resolution (width, height)
            use_gpu: Whether to use GPU acceleration if available
            track_thresh: Detection threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Setup ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=30
        )
        
        # Track history for trajectory visualization
        self.track_history = defaultdict(lambda: [])
        self.max_history_length = 50
        
        # Initialize annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=50,
            position=sv.Position.BOTTOM_CENTER
        )
        
        print(f"ONNX model loaded: {model_path}")
        print(f"Provider: {self.session.get_providers()[0]}")
        print("ByteTrack tracker initialized")

    def preprocess(self, image):
        """
        Preprocess image for model input.
        """
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(image)
        
        orig_h, orig_w = image_rgb.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, (orig_w, orig_h)

    def postprocess(self, outputs, orig_size, conf_threshold=None):
        """
        Postprocess model outputs to get detections.
        """
        conf_threshold = conf_threshold or self.confidence_threshold
        orig_w, orig_h = orig_size
        
        # Parse RF-DETR outputs
        boxes = outputs[0]  # [1, 300, 4] - normalized [cx, cy, w, h]
        logits = outputs[1]  # [1, 300, 91] - raw logits
        
        # Remove batch dimension
        boxes = boxes[0]
        logits = logits[0]
        
        # Apply sigmoid to convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        # Get max probability and corresponding class
        scores = np.max(probs, axis=1)
        labels = np.argmax(probs, axis=1)
        
        # Filter by confidence threshold
        conf_mask = scores >= conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        labels = labels[conf_mask]
        
        if len(boxes) == 0:
            return sv.Detections.empty()
        
        # Convert boxes from normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # Filter for vehicle classes only
        vehicle_mask = np.isin(labels, VEHICLE_CLASS_IDS)
        boxes_xyxy = boxes_xyxy[vehicle_mask]
        scores = scores[vehicle_mask]
        labels = labels[vehicle_mask]
        
        if len(boxes_xyxy) == 0:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=boxes_xyxy.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int)
        )

    def detect(self, image):
        """
        Perform vehicle detection on an image.
        """
        input_tensor, orig_size = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, orig_size)
        return detections

    def detect_and_track(self, image):
        """
        Perform vehicle detection and tracking on an image.
        
        Args:
            image: numpy array (BGR)
            
        Returns:
            supervision.Detections object with tracker_id field populated
        """
        # Detect vehicles
        detections = self.detect(image)
        
        if len(detections) == 0:
            return detections
        
        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Update track history for trajectory visualization
        if tracked_detections.tracker_id is not None:
            for tracker_id, box in zip(tracked_detections.tracker_id, tracked_detections.xyxy):
                # Get center point of bounding box
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # Add to history
                self.track_history[tracker_id].append((center_x, center_y))
                
                # Limit history length
                if len(self.track_history[tracker_id]) > self.max_history_length:
                    self.track_history[tracker_id].pop(0)
        
        return tracked_detections

    def annotate(self, image, detections, show_trace=True):
        """
        Draw bounding boxes, labels, and trajectories on the image.
        
        Args:
            image: numpy array (BGR)
            detections: supervision.Detections object with tracker_id
            show_trace: Whether to show movement trajectory
            
        Returns:
            Annotated image
        """
        if len(detections) == 0:
            return image.copy()
        
        # Generate labels with track ID
        labels = []
        if detections.tracker_id is not None:
            for tracker_id, class_id, conf in zip(
                detections.tracker_id, detections.class_id, detections.confidence
            ):
                class_name = VEHICLE_CLASS_NAMES.get(class_id, 'vehicle')
                labels.append(f"#{tracker_id} {class_name} {conf:.2f}")
        else:
            for class_id, conf in zip(detections.class_id, detections.confidence):
                class_name = VEHICLE_CLASS_NAMES.get(class_id, 'vehicle')
                labels.append(f"{class_name} {conf:.2f}")
        
        # Annotate image
        annotated = image.copy()
        
        # Draw trajectory traces
        if show_trace and detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(annotated, detections)
        
        # Draw bounding boxes and labels
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        return annotated

    def draw_trajectories(self, image, detections):
        """
        Draw custom movement trajectories on the image.
        
        Args:
            image: numpy array (BGR)
            detections: supervision.Detections object with tracker_id
            
        Returns:
            Image with trajectories drawn
        """
        if detections.tracker_id is None:
            return image
        
        for tracker_id in detections.tracker_id:
            if tracker_id in self.track_history and len(self.track_history[tracker_id]) > 1:
                points = self.track_history[tracker_id]
                
                # Draw trajectory line
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    
                    # Fade color based on age (older = more transparent)
                    alpha = i / len(points)
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                    thickness = max(1, int(2 * alpha))
                    
                    cv2.line(image, pt1, pt2, color, thickness)
        
        return image

    def reset_tracker(self):
        """Reset the tracker state and track history."""
        self.tracker.reset()
        self.track_history.clear()
        print("Tracker reset")

    def process_video(self, video_path, output_path=None, show=False, show_trace=True):
        """
        Process a video with vehicle detection and tracking.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show: Whether to display the video during processing
            show_trace: Whether to show movement trajectories
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Reset tracker for new video
        self.reset_tracker()
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Update tracker frame rate
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=fps
        )
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_tracks = set()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect and track vehicles
                detections = self.detect_and_track(frame)
                
                # Collect unique track IDs
                if detections.tracker_id is not None:
                    total_tracks.update(detections.tracker_id.tolist())
                
                # Annotate frame
                annotated = self.annotate(frame, detections, show_trace)
                
                # Add stats overlay
                cv2.putText(
                    annotated,
                    f"Frame: {frame_count}/{total_frames} | Active: {len(detections)} | Total tracks: {len(total_tracks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Write/display frame
                if writer:
                    writer.write(annotated)
                
                if show:
                    cv2.imshow("Vehicle Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames | Active tracks: {len(detections)} | Total: {len(total_tracks)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total unique vehicles tracked: {len(total_tracks)}")
        if output_path:
            print(f"Saved video to: {output_path}")
        
        return total_tracks

    def process_webcam(self, camera_id=0, show_trace=True):
        """
        Run real-time vehicle tracking on webcam feed.
        
        Args:
            camera_id: Camera device ID
            show_trace: Whether to show movement trajectories
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Reset tracker
        self.reset_tracker()
        
        print("Press 'q' to quit, 'r' to reset tracker")
        
        total_tracks = set()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.detect_and_track(frame)
                
                if detections.tracker_id is not None:
                    total_tracks.update(detections.tracker_id.tolist())
                
                annotated = self.annotate(frame, detections, show_trace)
                
                # Display stats
                cv2.putText(
                    annotated,
                    f"Active: {len(detections)} | Total: {len(total_tracks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Vehicle Tracking", annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_tracker()
                    total_tracks.clear()
                    print("Tracker reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"Total unique vehicles tracked: {len(total_tracks)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Vehicle Detection + Tracking")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to video or 'webcam' for live feed")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for annotated video")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[384, 384],
                        help="Model input size (width height)")
    parser.add_argument("--show", action="store_true",
                        help="Display output during processing")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable trajectory visualization")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    
    # Tracker parameters
    parser.add_argument("--track-thresh", type=float, default=0.25,
                        help="Tracking activation threshold")
    parser.add_argument("--track-buffer", type=int, default=30,
                        help="Frames to keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                        help="IOU matching threshold")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = VehicleTracker(
        model_path=args.model,
        confidence_threshold=args.confidence,
        input_size=tuple(args.input_size),
        use_gpu=not args.cpu,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh
    )
    
    # Process based on source type
    show_trace = not args.no_trace
    
    if args.source.lower() == "webcam":
        tracker.process_webcam(show_trace=show_trace)
    else:
        tracker.process_video(args.source, args.output, args.show, show_trace)


if __name__ == "__main__":
    main()
