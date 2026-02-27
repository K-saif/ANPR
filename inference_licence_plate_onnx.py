import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image


class RFDETRInference:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the ONNX inference session.
        
        Args:
            model_path: Path to the exported ONNX model
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"Model input: {self.input_name}, shape: {self.input_shape}")
        
        # Get model output details
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Model outputs: {self.output_names}")
    
    def preprocess(self, image_path):
        """
        Preprocess image for RF-DETR inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor and original image
        """
        # Load image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.original_size = original_image.shape[:2]  # (height, width)
        
        # Get target size from model input shape (typically [1, 3, H, W])
        if len(self.input_shape) == 4:
            target_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 560
            target_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 560
        else:
            target_h, target_w = 560, 560  # Default RF-DETR input size
        
        # Resize image
        image = cv2.resize(original_image, (target_w, target_h))
        
        # Normalize to [0, 1] and convert to float32
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert HWC to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image, original_image
    
    def postprocess(self, outputs, original_image):
        """
        Post-process model outputs to get bounding boxes and scores.
        
        Args:
            outputs: Raw model outputs
            original_image: Original input image for scaling boxes
            
        Returns:
            List of detections with boxes, scores, and class IDs
        """
        detections = []
        orig_h, orig_w = self.original_size
        
        # RF-DETR typically outputs: boxes (normalized), scores, labels
        # The exact format depends on the export settings
        if len(outputs) >= 2:
            # Common format: [boxes, scores] or [boxes, scores, labels]
            boxes = outputs[0]  # Shape: [batch, num_queries, 4] - usually in cxcywh format
            scores = outputs[1]  # Shape: [batch, num_queries, num_classes] or [batch, num_queries]
            
            # Handle different output formats
            if len(boxes.shape) == 3:
                boxes = boxes[0]  # Remove batch dimension
            if len(scores.shape) == 3:
                # Get max score and class for each detection
                class_ids = np.argmax(scores[0], axis=1)
                scores = np.max(scores[0], axis=1)
            else:
                scores = scores[0] if len(scores.shape) > 1 else scores
                class_ids = np.zeros(len(scores), dtype=int)
            
            # Filter by confidence threshold
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
                cx, cy, w, h = box
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                
                # Clip to image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'score': float(score),
                    'class_id': int(class_id)
                })
        
        return detections
    
    def predict(self, image_path):
        """
        Run inference on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections and the original image
        """
        # Preprocess
        input_tensor, original_image = self.preprocess(image_path)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, original_image)
        
        return detections, original_image
    
    def visualize(self, image, detections, output_path=None):
        """
        Draw detections on image.
        
        Args:
            image: Original RGB image
            detections: List of detection dictionaries
            output_path: Optional path to save the result
            
        Returns:
            Image with drawn detections
        """
        image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        if output_path:
            # Convert RGB to BGR for saving with cv2
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Result saved to: {output_path}")
        
        return image


    def preprocess_frame(self, frame):
        """
        Preprocess a video frame (already loaded as numpy array).
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.original_size = frame_rgb.shape[:2]  # (height, width)
        
        # Get target size from model input shape
        if len(self.input_shape) == 4:
            target_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 560
            target_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 560
        else:
            target_h, target_w = 560, 560
        
        # Resize image
        image = cv2.resize(frame_rgb, (target_w, target_h))
        
        # Normalize to [0, 1] and convert to float32
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert HWC to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image, frame_rgb
    
    def predict_frame(self, frame):
        """
        Run inference on a video frame.
        
        Args:
            frame: Input frame in BGR format (from cv2.VideoCapture)
            
        Returns:
            List of detections and the RGB frame
        """
        # Preprocess
        input_tensor, frame_rgb = self.preprocess_frame(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, frame_rgb)
        
        return detections, frame_rgb
    
    def process_video(self, video_path, output_path=None, show=True):
        """
        Process a video file and detect licence plates in each frame.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            show: Whether to display the video while processing
            
        Returns:
            Total number of frames processed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            detections, frame_rgb = self.predict_frame(frame)
            
            # Draw detections on frame
            result_frame = self.visualize(frame_rgb, detections)
            
            # Convert back to BGR for display/saving
            result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            
            # Write to output video
            if writer:
                writer.write(result_bgr)
            
            # Display
            if show:
                # Add frame info
                info = f"Frame: {frame_count}/{total_frames} | Plates: {len(detections)}"
                cv2.putText(result_bgr, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Licence Plate Detection", result_bgr)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        if writer:
            writer.release()
            print(f"Output saved to: {output_path}")
        if show:
            cv2.destroyAllWindows()
        
        print(f"Completed! Processed {frame_count} frames.")
        return frame_count


if __name__ == "__main__":
    # Path to your exported ONNX model
    model_path = "/home/medprime/Music/ANPR/output/licence_plate_inference_model.onnx"
    
    # Initialize inference
    detector = RFDETRInference(model_path, confidence_threshold=0.5)
    
    # ===== VIDEO INFERENCE =====
    video_path = "/home/medprime/Music/ANPR/anpr-demo-video.mp4"  # Change to your video
    output_path = "/home/medprime/Music/ANPR/licence_plt_output_video.mp4"  # Output video path
    
    # Process video (set show=False to disable live display)
    detector.process_video(video_path, output_path=output_path, show=True)
    
    # ===== IMAGE INFERENCE (uncomment to use) =====
    # image_path = "/home/medprime/Music/ANPR/test_image.jpg"
    # detections, original_image = detector.predict(image_path)
    # print(f"\nFound {len(detections)} licence plate(s):")
    # for i, det in enumerate(detections):
    #     print(f"  {i+1}. Box: {det['box']}, Score: {det['score']:.3f}")
    # detector.visualize(original_image, detections, "result.jpg")
