from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)


# ============================================================================
# Image Captioning
# ============================================================================

def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )
    captioning_model.to(device)

    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        if prompt is None:
            # Perform VQA
            inputs = captioning_processor(
                images=images, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            # Regular captioning. Prompt is a list of strings, one for each image
            assert len(images) == len(
                prompt
            ), "Number of images and prompts must match, got {} and {}".format(
                len(images), len(prompt)
            )
            inputs = captioning_processor(
                images=images, text=prompt, return_tensors="pt"
            ).to(device, dtype)
            generated_ids = captioning_model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            captions = captioning_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return captions

    return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score


# ============================================================================
# OCR Utilities for TVSC
# ============================================================================

def get_ocr_fn(use_gpu: bool = False, lang: str = 'en') -> callable:
    """
    Get an OCR function for text extraction from images.
    
    Args:
        use_gpu: Whether to use GPU for OCR
        lang: Language for OCR ('en', 'ch', etc.)
    
    Returns:
        OCR function that takes an image and returns text regions
    """
    try:
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
        
        def ocr_fn(
            image: Image.Image,
            return_boxes: bool = False
        ) -> List[dict]:
            """
            Extract text from image using OCR.
            
            Args:
                image: PIL Image to process
                return_boxes: Whether to return bounding boxes
            
            Returns:
                List of dicts with 'text', 'confidence', and optionally 'bbox'
            """
            img_array = np.array(image)
            results = ocr.ocr(img_array, cls=True)
            
            output = []
            if results and results[0]:
                for line in results[0]:
                    polygon = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    result = {
                        'text': text,
                        'confidence': confidence
                    }
                    
                    if return_boxes:
                        x_coords = [p[0] for p in polygon]
                        y_coords = [p[1] for p in polygon]
                        result['bbox'] = {
                            'x1': int(min(x_coords)),
                            'y1': int(min(y_coords)),
                            'x2': int(max(x_coords)),
                            'y2': int(max(y_coords))
                        }
                        result['polygon'] = polygon
                    
                    output.append(result)
            
            return output
        
        return ocr_fn
        
    except ImportError:
        print("[Warning] PaddleOCR not installed. Using fallback OCR.")
        
        # Fallback: try pytesseract
        try:
            import pytesseract
            
            def ocr_fn(
                image: Image.Image,
                return_boxes: bool = False
            ) -> List[dict]:
                """Fallback OCR using pytesseract."""
                if return_boxes:
                    data = pytesseract.image_to_data(
                        image, output_type=pytesseract.Output.DICT
                    )
                    output = []
                    for i, text in enumerate(data['text']):
                        if text.strip():
                            result = {
                                'text': text,
                                'confidence': data['conf'][i] / 100.0
                            }
                            result['bbox'] = {
                                'x1': data['left'][i],
                                'y1': data['top'][i],
                                'x2': data['left'][i] + data['width'][i],
                                'y2': data['top'][i] + data['height'][i]
                            }
                            output.append(result)
                    return output
                else:
                    text = pytesseract.image_to_string(image)
                    return [{'text': text.strip(), 'confidence': 1.0}]
            
            return ocr_fn
            
        except ImportError:
            print("[Warning] No OCR backend available.")
            
            def ocr_fn(image: Image.Image, return_boxes: bool = False) -> List[dict]:
                return []
            
            return ocr_fn


def extract_text_from_image(
    image: Image.Image,
    ocr_fn: Optional[callable] = None,
    use_gpu: bool = False
) -> str:
    """
    Extract all text from an image.
    
    Args:
        image: PIL Image
        ocr_fn: Optional pre-initialized OCR function
        use_gpu: Whether to use GPU (only used if ocr_fn is None)
    
    Returns:
        Extracted text as a single string
    """
    if ocr_fn is None:
        ocr_fn = get_ocr_fn(use_gpu=use_gpu)
    
    results = ocr_fn(image, return_boxes=False)
    return ' '.join([r['text'] for r in results])


def extract_text_regions(
    image: Image.Image,
    ocr_fn: Optional[callable] = None,
    use_gpu: bool = False
) -> List[dict]:
    """
    Extract text regions with bounding boxes from an image.
    
    Args:
        image: PIL Image
        ocr_fn: Optional pre-initialized OCR function
        use_gpu: Whether to use GPU (only used if ocr_fn is None)
    
    Returns:
        List of dicts with 'text', 'confidence', 'bbox'
    """
    if ocr_fn is None:
        ocr_fn = get_ocr_fn(use_gpu=use_gpu)
    
    return ocr_fn(image, return_boxes=True)


def compute_text_overlap(
    bbox1: dict,
    bbox2: dict
) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        bbox1: Dict with 'x1', 'y1', 'x2', 'y2'
        bbox2: Dict with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        IoU score between 0 and 1
    """
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def crop_element_from_screenshot(
    screenshot: Image.Image,
    bbox: dict,
    padding: int = 0
) -> Optional[Image.Image]:
    """
    Crop an element from a screenshot given its bounding box.
    
    Args:
        screenshot: Full page screenshot
        bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2'
        padding: Optional padding around the element
    
    Returns:
        Cropped image or None if invalid
    """
    try:
        x1 = max(0, bbox['x1'] - padding)
        y1 = max(0, bbox['y1'] - padding)
        x2 = min(screenshot.width, bbox['x2'] + padding)
        y2 = min(screenshot.height, bbox['y2'] + padding)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return screenshot.crop((x1, y1, x2, y2))
    except Exception:
        return None
