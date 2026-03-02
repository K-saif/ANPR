"""ANPR - Vehicle Detection + Tracking + License Plate Recognition"""

import argparse
from processor import VideoProcessor
from config import DEFAULT_MODEL_PATH, DEFAULT_PLATE_MODEL_PATH, CROPPED_FOLDER


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Tracking + Plate Detection")
    
    # Source/output
    p.add_argument("--source", required=True, help="Video path or 'webcam'")
    p.add_argument("--output", help="Output video path")
    p.add_argument("--show", type=lambda x: x.lower() == 'true', default=False, help="Display output (true/false)")
    
    # Model settings
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Vehicle model path")
    p.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL_PATH, help="Plate model path")
    p.add_argument("--confidence", type=float, default=0.5, help="Vehicle confidence")
    p.add_argument("--plate-confidence", type=float, default=0.2, help="Plate confidence")
    p.add_argument("--input-size", type=int, nargs=2, default=[384, 384], help="Input size")
    p.add_argument("--gpu", type=lambda x: x.lower() == 'true', default=False, help="Use GPU acceleration (true/false)")
    
    # Tracking
    p.add_argument("--track-thresh", type=float, default=0.25)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--no-trace", action="store_true", help="Disable trajectories")
    
    # Plate detection
    p.add_argument("--no-plates", action="store_true", help="Disable plate detection")
    p.add_argument("--cropped-folder", default=CROPPED_FOLDER)
    p.add_argument("--max-plates", type=int, default=3, help="Max plates per vehicle")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    processor = VideoProcessor(
        model_path=args.model,
        confidence=args.confidence,
        input_size=tuple(args.input_size),
        use_gpu=args.gpu,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        plate_model=args.plate_model,
        plate_confidence=args.plate_confidence,
        enable_plates=not args.no_plates,
        cropped_folder=args.cropped_folder,
        max_plate_detections=args.max_plates
    )
    
    if args.source.lower() == "webcam":
        processor.process_webcam(show_trace=not args.no_trace)
    else:
        processor.process_video(args.source, args.output, args.show, not args.no_trace)


if __name__ == "__main__":
    main()
