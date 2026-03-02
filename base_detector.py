"""Base ONNX detector with shared preprocessing logic."""

import cv2
import numpy as np
import onnxruntime as ort


class BaseDetector:
    """Base class for ONNX-based object detectors."""
    
    def __init__(self, model_path, confidence_threshold=0.5,
                 input_size=(384, 384), use_gpu=False):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Select providers based on GPU flag and availability
        if use_gpu:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("Warning: CUDA not available, falling back to CPU")
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Model loaded: {model_path}")
        print(f"Provider: {self.session.get_providers()[0]}")

    def preprocess(self, image, normalize_imagenet=False):
        """Convert image to model input tensor."""
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(image)
        
        orig_h, orig_w = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, self.input_size)
        normalized = resized.astype(np.float32) / 255.0
        
        if normalize_imagenet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
        
        batched = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        return batched, (orig_w, orig_h)

    def run_inference(self, input_tensor):
        """Run model inference."""
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def detect(self, image):
        """Override in subclass."""
        raise NotImplementedError
