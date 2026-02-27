"""
ONNX Vehicle Detection + Tracking Script
Uses RF-DETR ONNX model with ByteTrack for vehicle tracking
+ License Plate Detection for vehicles in bottom half of frame
"""

import cv2
import numpy as np
import os
from PIL import Image
import onnxruntime as ort
import supervision as sv
from collections import defaultdict

# Vehicle class IDs from COCO dataset
VEHICLE_CLASS_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# Default ONNX model paths
DEFAULT_MODEL_PATH = "/home/medprime/Music/ANPR/output/inference_model.onnx"
DEFAULT_PLATE_MODEL_PATH = "/home/medprime/Music/ANPR/output/licence_plate_inference_model.onnx"

# License plate detection settings
MAX_PLATE_DETECTIONS_PER_VEHICLE = 3  # Max times to detect plate per vehicle
CROPPED_FOLDER = "/home/medprime/Music/ANPR/cropped"


class LicensePlateDetector:
    """ONNX-based license plate detector using fine-tuned RF-DETR model."""
    
    def __init__(self, model_path=DEFAULT_PLATE_MODEL_PATH, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=True):
        """
        Initialize the license plate ONNX detector.
        
        Args:
            model_path: Path to license plate ONNX model
            confidence_threshold: Minimum confidence for detections
            input_size: Model input resolution (width, height)
            use_gpu: Whether to use GPU acceleration
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
        
        print(f"License plate model loaded: {model_path}")
    
    def preprocess(self, image):
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = np.array(image)
        
        orig_h, orig_w = image_rgb.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize to [0, 1] and apply ImageNet normalization
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Transpose to NCHW format and add batch dimension
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, (orig_w, orig_h)
    
    def postprocess(self, outputs, orig_size):
        """Postprocess model outputs to get license plate detections."""
        orig_w, orig_h = orig_size
        detections = []
        
        if len(outputs) >= 2:
            boxes = outputs[0]  # [1, num_queries, 4] - normalized [cx, cy, w, h]
            scores_output = outputs[1]  # [1, num_queries, num_classes] or [1, num_queries]
            
            # Remove batch dimension
            boxes = boxes[0]
            scores_output = scores_output[0]
            
            # Handle different output formats
            if len(scores_output.shape) == 2:
                # Multi-class output - get max score
                scores = np.max(scores_output, axis=1)
            else:
                scores = scores_output
            
            # Filter by confidence
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            for box, score in zip(boxes, scores):
                cx, cy, w, h = box
                x1 = int((cx - w / 2) * orig_w)
                y1 = int((cy - h / 2) * orig_h)
                x2 = int((cx + w / 2) * orig_w)
                y2 = int((cy + h / 2) * orig_h)
                
                # Clip to bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'score': float(score)
                    })
        
        return detections
    
    def detect(self, image):
        """
        Detect license plates in an image.
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            List of detection dicts with 'box' and 'score'
        """
        input_tensor, orig_size = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, orig_size)
        return detections


class VehicleTracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=True,
                 track_thresh=0.25, track_buffer=30, match_thresh=0.8,
                 plate_model_path=DEFAULT_PLATE_MODEL_PATH, plate_confidence=0.5,
                 enable_plate_detection=True, cropped_folder=CROPPED_FOLDER):
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
            plate_model_path: Path to license plate ONNX model
            plate_confidence: Confidence threshold for plate detection
            enable_plate_detection: Whether to enable plate detection
            cropped_folder: Folder to save cropped plate images
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.enable_plate_detection = enable_plate_detection
        self.cropped_folder = cropped_folder
        
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
        
        # License plate detection tracking
        self.plate_detection_count = defaultdict(int)  # track_id -> detection count
        self.max_plate_detections = MAX_PLATE_DETECTIONS_PER_VEHICLE
        
        # Initialize license plate detector
        self.plate_detector = None
        if enable_plate_detection:
            try:
                self.plate_detector = LicensePlateDetector(
                    model_path=plate_model_path,
                    confidence_threshold=plate_confidence,
                    use_gpu=use_gpu
                )
                # Create cropped folder if it doesn't exist
                os.makedirs(self.cropped_folder, exist_ok=True)
                print(f"Cropped plates will be saved to: {self.cropped_folder}")
            except Exception as e:
                print(f"Warning: Could not load plate detector: {e}")
                self.enable_plate_detection = False
        
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

    def detect_plates_in_bottom_half(self, frame, detections, frame_number=0):
        """
        Detect license plates for vehicles in the bottom half of the frame.
        Limits detection to MAX_PLATE_DETECTIONS_PER_VEHICLE per vehicle.
        
        Args:
            frame: numpy array (BGR)
            detections: supervision.Detections object with tracker_id
            frame_number: Current frame number for filename
            
        Returns:
            List of plate detections with tracker_id info
        """
        if not self.enable_plate_detection or self.plate_detector is None:
            return []
        
        if detections.tracker_id is None or len(detections) == 0:
            return []
        
        frame_height = frame.shape[0]
        bottom_half_threshold = frame_height / 2  # Y coordinate threshold
        
        plate_results = []
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            
            # Check if vehicle center is in bottom half of frame
            vehicle_center_y = (y1 + y2) / 2
            if vehicle_center_y < bottom_half_threshold:
                continue  # Skip vehicles in top half
            
            # Check if we've already detected plates for this vehicle max times
            if self.plate_detection_count[tracker_id] >= self.max_plate_detections:
                continue  # Skip - already have enough detections for this vehicle
            
            # Crop vehicle from frame
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue
            
            # Detect license plates in vehicle crop
            plate_detections = self.plate_detector.detect(vehicle_crop)
            
            if plate_detections:
                # Increment detection count for this vehicle
                self.plate_detection_count[tracker_id] += 1
                count = self.plate_detection_count[tracker_id]
                
                for plate_det in plate_detections:
                    px1, py1, px2, py2 = plate_det['box']
                    
                    # Crop the plate from the vehicle crop
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    if plate_crop.size > 0:
                        # Save cropped plate image
                        filename = f"{tracker_id}_{count}_frame{frame_number}.jpg"
                        filepath = os.path.join(self.cropped_folder, filename)
                        cv2.imwrite(filepath, plate_crop)
                        
                        plate_results.append({
                            'tracker_id': tracker_id,
                            'plate_box': plate_det['box'],
                            'plate_score': plate_det['score'],
                            'vehicle_box': [x1, y1, x2, y2],
                            'saved_path': filepath
                        })
                        
                        print(f"Plate detected for vehicle #{tracker_id} "
                              f"(detection {count}/{self.max_plate_detections}) - saved: {filename}")
        
        return plate_results

    def reset_tracker(self):
        """Reset the tracker state, track history, and plate detection counts."""
        self.tracker.reset()
        self.track_history.clear()
        self.plate_detection_count.clear()
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
                
                # Detect license plates for vehicles in bottom half
                plate_results = self.detect_plates_in_bottom_half(frame, detections, frame_count)
                
                # Annotate frame
                annotated = self.annotate(frame, detections, show_trace)
                
                # Draw plate detection zone line (horizontal line at middle)
                cv2.line(annotated, (0, height // 2), (width, height // 2), (0, 255, 255), 2)
                cv2.putText(annotated, "Plate Detection Zone", (10, height // 2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Highlight vehicles with recent plate detections
                for plate_res in plate_results:
                    vx1, vy1, vx2, vy2 = plate_res['vehicle_box']
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                
                # Add stats overlay
                plates_saved = sum(self.plate_detection_count.values())
                cv2.putText(
                    annotated,
                    f"Frame: {frame_count}/{total_frames} | Vehicles: {len(detections)} | Plates saved: {plates_saved}",
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
                    plates_saved = sum(self.plate_detection_count.values())
                    print(f"Processed {frame_count}/{total_frames} frames | Vehicles: {len(detections)} | Plates saved: {plates_saved}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total unique vehicles tracked: {len(total_tracks)}")
        print(f"Total plate images saved: {sum(self.plate_detection_count.values())}")
        print(f"Vehicles with plates detected: {len(self.plate_detection_count)}")
        if output_path:
            print(f"Saved video to: {output_path}")
        if self.enable_plate_detection:
            print(f"Cropped plates saved to: {self.cropped_folder}")
        
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
    
    # License plate detection parameters
    parser.add_argument("--plate-model", type=str, default=DEFAULT_PLATE_MODEL_PATH,
                        help="Path to license plate ONNX model")
    parser.add_argument("--plate-confidence", type=float, default=0.2,
                        help="Confidence threshold for plate detection")
    parser.add_argument("--no-plate-detection", action="store_true",
                        help="Disable license plate detection")
    parser.add_argument("--cropped-folder", type=str, default=CROPPED_FOLDER,
                        help="Folder to save cropped plate images")
    parser.add_argument("--max-plate-detections", type=int, default=3,
                        help="Max plate detections per vehicle (default: 3)")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = VehicleTracker(
        model_path=args.model,
        confidence_threshold=args.confidence,
        input_size=tuple(args.input_size),
        use_gpu=not args.cpu,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        plate_model_path=args.plate_model,
        plate_confidence=args.plate_confidence,
        enable_plate_detection=not args.no_plate_detection,
        cropped_folder=args.cropped_folder
    )
    
    # Set max plate detections
    tracker.max_plate_detections = args.max_plate_detections
    
    # Process based on source type
    show_trace = not args.no_trace
    
    if args.source.lower() == "webcam":
        tracker.process_webcam(show_trace=show_trace)
    else:
        tracker.process_video(args.source, args.output, args.show, show_trace)


if __name__ == "__main__":
    main()
