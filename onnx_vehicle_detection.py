"""
ONNX Vehicle Detection Inference Script
Uses RF-DETR exported ONNX model for vehicle detection
"""

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import supervision as sv

# Vehicle class IDs from COCO dataset
VEHICLE_CLASS_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# Default ONNX model path
DEFAULT_MODEL_PATH = "/home/medprime/Music/ANPR/output/inference_model.onnx"


class ONNXVehicleDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.5, 
                 input_size=(384, 384), use_gpu=True):
        """
        Initialize the ONNX vehicle detector.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            input_size: Model input resolution (width, height)
            use_gpu: Whether to use GPU acceleration if available
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
        
        # Initialize annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        print(f"ONNX model loaded: {model_path}")
        print(f"Input: {self.input_name} {self.input_shape}")
        print(f"Outputs: {self.output_names}")
        print(f"Provider: {self.session.get_providers()[0]}")

    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            Preprocessed tensor, original image size
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
        
        # Transpose to NCHW format (batch, channels, height, width)
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, (orig_w, orig_h)

    def postprocess(self, outputs, orig_size, conf_threshold=None):
        """
        Postprocess model outputs to get detections.
        
        RF-DETR ONNX output format:
        - outputs[0]: boxes in [cx, cy, w, h] normalized format, shape (1, 300, 4)
        - outputs[1]: logits for 91 COCO classes, shape (1, 300, 91) - need sigmoid!
        
        Args:
            outputs: Raw model outputs
            orig_size: Original image size (width, height)
            conf_threshold: Confidence threshold (uses instance default if None)
            
        Returns:
            supervision.Detections object
        """
        conf_threshold = conf_threshold or self.confidence_threshold
        orig_w, orig_h = orig_size
        
        # Parse RF-DETR outputs
        boxes = outputs[0]  # [1, 300, 4] - normalized [cx, cy, w, h]
        logits = outputs[1]  # [1, 300, 91] - raw logits, need sigmoid
        
        # Remove batch dimension
        boxes = boxes[0]  # [300, 4]
        logits = logits[0]  # [300, 91]
        
        # Apply sigmoid to convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        
        # Get max probability and corresponding class for each query
        scores = np.max(probs, axis=1)  # [300]
        labels = np.argmax(probs, axis=1)  # [300]
        
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
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            supervision.Detections object with filtered vehicle detections
        """
        # Preprocess
        input_tensor, orig_size = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs, orig_size)
        
        return detections

    def annotate(self, image, detections):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: numpy array (BGR)
            detections: supervision.Detections object
            
        Returns:
            Annotated image
        """
        if len(detections) == 0:
            return image.copy()
        
        # Generate labels
        labels = [
            f"{VEHICLE_CLASS_NAMES.get(class_id, 'vehicle')} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        
        # Annotate image
        annotated = self.box_annotator.annotate(image.copy(), detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        return annotated

    def process_image(self, image_path, output_path=None):
        """
        Process a single image for vehicle detection.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            
        Returns:
            detections, annotated_image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        detections = self.detect(image)
        annotated = self.annotate(image, detections)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved annotated image to: {output_path}")
        
        print(f"Detected {len(detections)} vehicles")
        return detections, annotated

    def process_video(self, video_path, output_path=None, show=False):
        """
        Process a video for vehicle detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show: Whether to display the video during processing
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect vehicles
                detections = self.detect(frame)
                annotated = self.annotate(frame, detections)
                
                # Write/display frame
                if writer:
                    writer.write(annotated)
                
                if show:
                    cv2.imshow("Vehicle Detection (ONNX)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        if output_path:
            print(f"Saved video to: {output_path}")

    def process_webcam(self, camera_id=0):
        """
        Run real-time vehicle detection on webcam feed.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.detect(frame)
                annotated = self.annotate(frame, detections)
                
                # Display detection count
                cv2.putText(
                    annotated, 
                    f"Vehicles: {len(detections)}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow("Vehicle Detection (ONNX)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Vehicle Detection")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image/video or 'webcam' for live feed")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for annotated image/video")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[384, 384],
                        help="Model input size (width height)")
    parser.add_argument("--show", action="store_true",
                        help="Display output during processing")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (disable GPU)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ONNXVehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        input_size=tuple(args.input_size),
        use_gpu=not args.cpu
    )
    
    # Process based on source type
    if args.source.lower() == "webcam":
        detector.process_webcam()
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.process_video(args.source, args.output, args.show)
    else:
        detector.process_image(args.source, args.output)


if __name__ == "__main__":
    main()
