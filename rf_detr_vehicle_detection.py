"""
RF-DETR Vehicle Detection Inference Script
"""

import cv2
import numpy as np
from PIL import Image
import torch
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRNano
import supervision as sv

# Vehicle class IDs from COCO dataset
VEHICLE_CLASS_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}


class VehicleDetector:
    def __init__(self, model_size="nano", confidence_threshold=0.3, device=None):
        """
        Initialize the RF-DETR vehicle detector.
        
        Args:
            model_size: "nano", "base", or "large" for model selection
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on (cuda/cpu)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load RF-DETR model
        if model_size == "large":
            self.model = RFDETRLarge()
        elif model_size == "base":
            self.model = RFDETRBase()
        else:
            self.model = RFDETRNano()  # Default: fastest model (2.3ms, 384x384)
        
        # Initialize annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        print(f"RF-DETR {model_size} model loaded on {self.device}")

    def detect(self, image):
        """
        Perform vehicle detection on an image.
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            supervision.Detections object with filtered vehicle detections
        """
        # Convert BGR to RGB if numpy array
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Run inference
        detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
        print(f"class detections: {detections.class_id}")
        # Filter for vehicle classes only
        vehicle_mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        filtered_detections = detections[vehicle_mask]
        
        return filtered_detections

    def annotate(self, image, detections):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: numpy array (BGR)
            detections: supervision.Detections object
            
        Returns:
            Annotated image
        """
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
                    cv2.imshow("Vehicle Detection", annotated)
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
                
                # Display FPS and detection count
                cv2.putText(
                    annotated, 
                    f"Vehicles: {len(detections)}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow("Vehicle Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-DETR Vehicle Detection")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image/video or 'webcam' for live feed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for annotated image/video")
    parser.add_argument("--model", type=str, default="nano", choices=["nano", "base", "large"],
                        help="Model size: nano (fastest), base, or large")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--show", action="store_true",
                        help="Display output during processing")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VehicleDetector(
        model_size=args.model,
        confidence_threshold=args.confidence
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
