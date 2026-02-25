"""
License Plate Detection using YOLO ONNX Model
==============================================
This script provides inference capabilities for YOLO models converted to ONNX format.

Usage:
    # Detect on image
    python lpr_inference.py --model model.onnx --source image.jpg --output result.jpg
    
    # Detect on video
    python lpr_inference.py --model model.onnx --source video.mp4 --output result.mp4
    
    # Detect from webcam
    python lpr_inference.py --model model.onnx --source 0

Requirements:
    pip install opencv-python numpy onnxruntime
    # For GPU support:
    pip install onnxruntime-gpu
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time


class LicensePlateDetector:
    """YOLO ONNX model for license plate detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model
        providers = self._get_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Handle dynamic input dimensions (when shape contains strings like 'height', 'width')
        self.input_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.input_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640
        
        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape} (using {self.input_width}x{self.input_height})")
        print(f"Providers: {self.session.get_providers()}")
    
    def _get_providers(self):
        """Get available execution providers."""
        available = ort.get_available_providers()
        providers = []
        
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        
        return providers if providers else ['CPUExecutionProvider']
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for inference.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed image tensor and scale factors
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scale factors
        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height
        
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, (scale_x, scale_y)
    
    def postprocess(self, outputs: np.ndarray, scale_factors: tuple, original_shape: tuple) -> list:
        """
        Post-process model outputs.
        
        Args:
            outputs: Raw model outputs
            scale_factors: (scale_x, scale_y) for rescaling boxes
            original_shape: (height, width) of original image
            
        Returns:
            List of detections: [{'bbox': (x1,y1,x2,y2), 'confidence': float, 'class_id': int}, ...]
        """
        scale_x, scale_y = scale_factors
        orig_height, orig_width = original_shape
        
        # Handle different YOLO output formats
        predictions = outputs[0]
        
        # YOLOv8 format: (1, 84, 8400) -> transpose to (8400, 84)
        if len(predictions.shape) == 3:
            if predictions.shape[1] < predictions.shape[2]:
                predictions = np.transpose(predictions, (0, 2, 1))
            predictions = predictions[0]
        
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in predictions:
            # For YOLOv8: first 4 values are box coordinates, rest are class scores
            if len(detection) > 5:
                # YOLOv8 format: [x_center, y_center, width, height, class1_score, class2_score, ...]
                x_center, y_center, width, height = detection[:4]
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
            else:
                # YOLOv5 format: [x_center, y_center, width, height, obj_conf, class_conf]
                x_center, y_center, width, height, obj_conf = detection[:5]
                class_scores = detection[5:]
                class_id = np.argmax(class_scores) if len(class_scores) > 0 else 0
                confidence = obj_conf * (class_scores[class_id] if len(class_scores) > 0 else 1)
            
            if confidence < self.conf_threshold:
                continue
            
            # Convert to corner coordinates and scale to original image
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h for NMS
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
            
            detections = []
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, w, h = boxes[idx]
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': confidences[idx],
                    'class_id': class_ids[idx]
                })
            return detections
        
        return []
    
    def detect(self, image: np.ndarray) -> list:
        """
        Run detection on an image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of detections
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor, scale_factors = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, scale_factors, original_shape)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: list, label: str = "License Plate") -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: BGR image
            detections: List of detection dictionaries
            label: Label for the detections
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text = f"{label}: {conf:.2f}"
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            cv2.rectangle(result, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(result, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return result
    
    def crop_plates(self, image: np.ndarray, detections: list, padding: int = 5) -> list:
        """
        Crop detected license plates from image.
        
        Args:
            image: BGR image
            detections: List of detection dictionaries
            padding: Extra padding around the crop
            
        Returns:
            List of cropped plate images
        """
        crops = []
        h, w = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            crops.append({
                'image': crop,
                'bbox': (x1, y1, x2, y2),
                'confidence': det['confidence']
            })
        
        return crops


def detect_image(model_path: str, image_path: str, output_path: str = None, 
                 conf: float = 0.5, iou: float = 0.45, save_crops: bool = False):
    """Detect license plates in a single image."""
    detector = LicensePlateDetector(model_path, conf, iou)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = time.time() - start_time
    
    print(f"Found {len(detections)} license plate(s) in {inference_time*1000:.2f}ms")
    
    for i, det in enumerate(detections):
        print(f"  Detection {i+1}: bbox={det['bbox']}, confidence={det['confidence']:.3f}")
    
    result = detector.draw_detections(image, detections)
    
    # Save cropped plates if requested
    if save_crops and detections:
        crops = detector.crop_plates(image, detections)
        output_dir = Path(output_path).parent if output_path else Path(image_path).parent
        for i, crop in enumerate(crops):
            crop_path = output_dir / f"plate_{i+1}.jpg"
            cv2.imwrite(str(crop_path), crop['image'])
            print(f"Saved crop: {crop_path}")
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow("License Plate Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def detect_video(model_path: str, video_path: str, output_path: str = None,
                 conf: float = 0.5, iou: float = 0.45):
    """Detect license plates in a video."""
    detector = LicensePlateDetector(model_path, conf, iou)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        start_time = time.time()
        detections = detector.detect(frame)
        total_time += time.time() - start_time
        
        result = detector.draw_detections(frame, detections)
        
        # Add FPS counter
        avg_fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if writer:
            writer.write(result)
        else:
            cv2.imshow("License Plate Detection (Press 'q' to quit)", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, avg FPS: {avg_fps:.1f}")
    
    cap.release()
    if writer:
        writer.release()
        print(f"Result saved to {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Total: {frame_count} frames, avg inference time: {total_time/frame_count*1000:.2f}ms")


def detect_webcam(model_path: str, camera_id: int = 0, conf: float = 0.5, iou: float = 0.45):
    """Detect license plates from webcam in real-time."""
    detector = LicensePlateDetector(model_path, conf, iou)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        detections = detector.detect(frame)
        result = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        elapsed = time.time() - fps_start
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(result, f"FPS: {fps:.1f} | Detections: {len(detections)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("License Plate Detection (Press 'q' to quit)", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="License Plate Detection using YOLO ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--source", type=str, default="0", help="Image/video path or camera ID (default: 0 for webcam)")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped license plates")
    
    args = parser.parse_args()
    
    # Determine source type
    source = args.source
    
    if source.isdigit():
        # Webcam
        detect_webcam(args.model, int(source), args.conf, args.iou)
    elif Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
        # Image
        detect_image(args.model, source, args.output, args.conf, args.iou, args.save_crops)
    elif Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv']:
        # Video
        detect_video(args.model, source, args.output, args.conf, args.iou)
    else:
        print(f"Unknown source type: {source}")
        print("Supported: images (.jpg, .png), videos (.mp4, .avi), or camera ID (0, 1)")


if __name__ == "__main__":
    main()
