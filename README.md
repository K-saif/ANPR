# ANPR - Automatic Number Plate Recognition

A real-time vehicle detection, tracking, and license plate recognition system using ONNX models based on RF-DETR with PaddleOCR for Arabic and English text extraction.

## Features

- **Vehicle Detection**: RF-DETR based detection for cars, motorcycles, buses, and trucks
- **Multi-Object Tracking**: ByteTrack for robust vehicle tracking across frames
- **License Plate Detection**: Dedicated ONNX model for plate localization
- **Dual-Language OCR**: PaddleOCR with Arabic and English support
- **Parallel Processing**: Non-blocking OCR runs in background threads
- **Zone-Based Detection**: Plates detected only in configurable regions (bottom half by default)
- **CSV Export**: Automatic export of OCR results with timestamps

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ANPR.git
cd ANPR

# Create virtual environment
conda create -n anpr python=3.10
conda activate anpr

# Install dependencies (CPU)
pip install -r requirements.txt

# For GPU support
pip uninstall onnxruntime paddlepaddle
pip install onnxruntime-gpu paddlepaddle-gpu
```

## Usage

### Basic Usage

```bash
# Process video (CPU, no display)
python main.py --source video.mp4 --output output.mp4

# Process video with display
python main.py --source video.mp4 --output output.mp4 --show true

# Process video with GPU acceleration
python main.py --source video.mp4 --output output.mp4 --show true --gpu true

# Process webcam
python main.py --source webcam --show true
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | *required* | Video file path or `webcam` |
| `output` | None | Output video path |
| `show` | false | Display output window (true/false) |
| `gpu` | false | Use GPU acceleration (true/false) |
| `model` | `output/inference_model.onnx` | Vehicle detection model |
| `plate-model` | `output/licence_plate_inference_model.onnx` | Plate detection model |
| `confidence` | 0.5 | Vehicle detection confidence |
| `plate-confidence` | 0.2 | Plate detection confidence |
| `input-size` | 384 384 | Model input dimensions |
| `track-thresh` | 0.25 | Tracking activation threshold |
| `track-buffer` | 30 | Lost track buffer (frames) |
| `match-thresh` | 0.8 | Tracking match threshold |
| `no-trace` | false | Disable trajectory visualization |
| `no-plates` | false | Disable plate detection |
| `cropped-folder` | `cropped/` | Folder for plate crops |
| `max-plates` | 3 | Max plate detections per vehicle |

### Process Existing Cropped Plates

```bash
python plate_ocr.py
```

This processes all images in the `cropped/` folder and saves results to `ocr_results.csv`.

## Project Structure

```
ANPR/
├── main.py              # CLI entry point
├── processor.py         # Main video processing pipeline
├── vehicle_tracker.py   # Vehicle detection + ByteTrack tracking
├── plate_detector.py    # License plate detection
├── plate_ocr.py         # PaddleOCR wrapper (Arabic + English)
├── ocr_worker.py        # Parallel OCR processing
├── base_detector.py     # Shared ONNX inference logic
├── visualizer.py        # Annotation and display utilities
├── config.py            # Configuration constants
├── requirements.txt     # Dependencies
├── output/              # ONNX models
│   ├── inference_model.onnx
│   └── licence_plate_inference_model.onnx
└── cropped/             # Saved plate crops + OCR results
    └── ocr_results.csv
```

## Output

### Cropped Plate Images

Saved to `cropped/` folder with naming convention:
```
{tracker_id}_{detection_count}_f{frame_number}.jpg
```
Example: `51_2_f153.jpg` (Tracker #51, 2nd detection, frame 153)

### OCR Results CSV

`cropped/ocr_results.csv` contains:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp |
| frame_num | Video frame number |
| tracker_id | Vehicle tracking ID |
| plate_text | Combined OCR result |
| plate_text_en | English OCR result |
| plate_text_ar | Arabic OCR result |
| confidence | OCR confidence score |
| vehicle_box | Vehicle bounding box [x1,y1,x2,y2] |
| image_path | Path to cropped plate image |

## Configuration

Edit `config.py` to modify:

```python
# Vehicle classes to detect (COCO IDs)
VEHICLE_CLASS_IDS = [3, 4, 6, 8]  # car, motorcycle, bus, truck

# Model paths
DEFAULT_MODEL_PATH = "output/inference_model.onnx"
DEFAULT_PLATE_MODEL_PATH = "output/licence_plate_inference_model.onnx"

# Detection settings
MAX_PLATE_DETECTIONS_PER_VEHICLE = 3
CROPPED_FOLDER = "cropped"
```

## Requirements

- Python 3.10+
- OpenCV
- PyTorch
- ONNX Runtime
- PaddlePaddle + PaddleOCR
- Supervision (ByteTrack)
- RF-DETR

See `requirements.txt` for full dependencies.

## Workflow

```
Video Frame
    │
    ▼
┌─────────────────┐
│Vehicle Detection│  (RF-DETR ONNX)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ByteTrack      │  (Multi-object tracking)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Zone Filter     │  (Bottom half of frame)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Plate Detection │  (ONNX, max 3 per vehicle)
└────────┬────────┘
         │
         ├──► Save cropped image
         │
         ▼
┌─────────────────┐
│ Parallel OCR    │  (PaddleOCR: Arabic + English)
└────────┬────────┘
         │
         ▼
    CSV Results
```

## License

MIT License

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) - Vehicle detection
- [Supervision](https://github.com/roboflow/supervision) - ByteTrack tracking
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Text recognition
