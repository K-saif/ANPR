"""License plate detector using ONNX model."""

import numpy as np
from base_detector import BaseDetector
from config import DEFAULT_PLATE_MODEL_PATH


class PlateDetector(BaseDetector):
    """Detects license plates in images."""
    
    def __init__(self, model_path=DEFAULT_PLATE_MODEL_PATH, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=True):
        super().__init__(model_path, confidence_threshold, input_size, use_gpu)

    def detect(self, image):
        """
        Detect license plates.
        
        Returns:
            List of dicts with 'box' [x1,y1,x2,y2] and 'score'
        """
        input_tensor, (orig_w, orig_h) = self.preprocess(image, normalize_imagenet=True)
        outputs = self.run_inference(input_tensor)
        
        boxes, scores_out = outputs[0][0], outputs[1][0]
        
        # Handle multi-class output
        scores = np.max(scores_out, axis=1) if len(scores_out.shape) == 2 else scores_out
        
        detections = []
        for box, score in zip(boxes, scores):
            if score < self.confidence_threshold:
                continue
            
            cx, cy, w, h = box
            x1 = int(max(0, (cx - w/2) * orig_w))
            y1 = int(max(0, (cy - h/2) * orig_h))
            x2 = int(min(orig_w, (cx + w/2) * orig_w))
            y2 = int(min(orig_h, (cy + h/2) * orig_h))
            
            if x2 > x1 and y2 > y1:
                detections.append({'box': [x1, y1, x2, y2], 'score': float(score)})
        
        return detections
