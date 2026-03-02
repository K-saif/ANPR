"""
PaddleOCR-based license plate text recognition.
Supports both Arabic and English text extraction.
"""

import os
# Disable PIR mode and MKL-DNN to fix compatibility issues
# Must be set BEFORE importing paddle
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
paddle.set_flags({'FLAGS_use_mkldnn': 0})

from paddleocr import PaddleOCR
import numpy as np
from typing import Optional, List, Tuple
import cv2


class PlateOCR:
    """OCR engine for license plate text recognition using PaddleOCR."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize PaddleOCR engines for Arabic and English.
        PaddleOCR requires separate runs for each language.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: False for CPU)
        """
        # English OCR engine
        self.ocr_en = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            use_gpu=use_gpu
            # show_log=False
        )
        
        # Arabic OCR engine
        self.ocr_ar = PaddleOCR(
            use_textline_orientation=True,
            lang='ar',
            use_gpu=use_gpu
            # show_log=False
        )
    
    def preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR results.
        
        Args:
            image: BGR plate image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Convert back to BGR for PaddleOCR
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def extract_text(
        self, 
        image: np.ndarray, 
        preprocess: bool = True
    ) -> dict:
        """
        Extract text from license plate image using both Arabic and English OCR.
        
        Args:
            image: BGR plate image (numpy array)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing:
                - 'text': Combined recognized text
                - 'english': English text results
                - 'arabic': Arabic text results
                - 'confidence': Average confidence score
                - 'details': Raw OCR details
        """
        if image is None or image.size == 0:
            return {
                'text': '',
                'english': '',
                'arabic': '',
                'confidence': 0.0,
                'details': []
            }
        
        # Preprocess if requested
        img_to_process = self.preprocess_plate(image) if preprocess else image
        
        # Run English OCR
        en_results = self._run_ocr(self.ocr_en, img_to_process)
        
        # Run Arabic OCR
        ar_results = self._run_ocr(self.ocr_ar, img_to_process)
        
        # Combine results
        combined_text = self._combine_results(en_results, ar_results)
        
        # Calculate average confidence
        all_confidences = [r[1] for r in en_results + ar_results]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            'text': combined_text,
            'english': ' '.join([r[0] for r in en_results]),
            'arabic': ' '.join([r[0] for r in ar_results]),
            'confidence': round(avg_confidence, 3),
            'details': {
                'english_results': en_results,
                'arabic_results': ar_results
            }
        }
    
    def _run_ocr(
        self, 
        ocr_engine: PaddleOCR, 
        image: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Run OCR engine on image.
        
        Args:
            ocr_engine: PaddleOCR instance
            image: Preprocessed image
            
        Returns:
            List of (text, confidence) tuples
        """
        results = []
        try:
            ocr_output = ocr_engine.ocr(image)
            
            if ocr_output and ocr_output[0]:
                for line in ocr_output[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if text.strip():
                            results.append((text.strip(), confidence))
        except Exception as e:
            print(f"OCR error: {e}")
        
        return results
    
    def _combine_results(
        self, 
        en_results: List[Tuple[str, float]], 
        ar_results: List[Tuple[str, float]]
    ) -> str:
        """
        Combine English and Arabic results intelligently.
        
        For license plates, typically the format includes both numbers/letters
        and potentially Arabic text. We prioritize higher confidence results.
        
        Args:
            en_results: English OCR results
            ar_results: Arabic OCR results
            
        Returns:
            Combined text string
        """
        # If one is empty, return the other
        if not en_results and not ar_results:
            return ''
        if not en_results:
            return ' '.join([r[0] for r in ar_results])
        if not ar_results:
            return ' '.join([r[0] for r in en_results])
        
        # Use the result with higher average confidence
        en_avg = sum([r[1] for r in en_results]) / len(en_results)
        ar_avg = sum([r[1] for r in ar_results]) / len(ar_results)
        
        # If confidences are close, combine both
        if abs(en_avg - ar_avg) < 0.1:
            en_text = ' '.join([r[0] for r in en_results])
            ar_text = ' '.join([r[0] for r in ar_results])
            return f"{en_text} | {ar_text}"
        
        # Return higher confidence result
        if en_avg > ar_avg:
            return ' '.join([r[0] for r in en_results])
        else:
            return ' '.join([r[0] for r in ar_results])
    
    def extract_text_from_file(self, image_path: str) -> dict:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to plate image file
            
        Returns:
            OCR results dictionary
        """
        image = cv2.imread(image_path)
        if image is None:
            return {
                'text': '',
                'english': '',
                'arabic': '',
                'confidence': 0.0,
                'details': [],
                'error': f'Failed to load image: {image_path}'
            }
        
        return self.extract_text(image)

