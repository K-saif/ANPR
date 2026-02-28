"""
Shared configuration constants for ANPR system
"""

# Vehicle class IDs from COCO dataset
VEHICLE_CLASS_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck
VEHICLE_CLASS_NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

# Default ONNX model paths
DEFAULT_MODEL_PATH = "/home/medprime/Music/ANPR/output/inference_model.onnx"
DEFAULT_PLATE_MODEL_PATH = "/home/medprime/Music/ANPR/output/licence_plate_inference_model.onnx"

# License plate detection settings
MAX_PLATE_DETECTIONS_PER_VEHICLE = 3  # Max times to detect plate per vehicle
CROPPED_FOLDER = "/home/medprime/Music/ANPR/cropped"
