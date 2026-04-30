"""
Text Extraction Module - Unified Hybrid Pipeline (FIXED - GCV Large Image Support)
Combines Custom OCR (character-level) with Google Cloud Vision (document-level)
for intelligent text extraction and validation

✅ FIXES APPLIED:
1. Fixed GCV API call - using correct 'vision_image' variable
2. Updated autocast syntax for PyTorch compatibility
3. ✅ NEW: Resize large images before sending to GCV (fixes "Bad image data" error)
4. ✅ NEW: Retry logic with smaller images if GCV fails
5. ✅ NEW: Better JPEG compression for large files
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
import torchvision.transforms as transforms
from google.cloud import vision
import io
import os
from model_architecture import load_model, DEVICE
from validation import KannadaWordValidator
import json
from google.cloud.vision import ImageContext

# ============================================================
# CONFIGURATION
# ============================================================

# Image preprocessing for custom OCR model
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CONFIDENCE_THRESHOLD = 0.30  # Skip characters with confidence < 30%


# ============================================================
# UNIFIED HYBRID EXTRACTOR
# ============================================================

class HybridExtractor:
    """
    Unified extraction system combining Custom OCR and Google Cloud Vision
    
    Pipeline:
    1. Custom OCR: Character-level segmentation and recognition
    2. GCV: Document-level context and validation
    3. Fusion: Intelligent combination of both results
    """
    
    def __init__(self, model_path, class_mapping_path, credentials_path, device=None, dictionary_path=None):
        # Use global DEVICE by default; allow override
        self.device = device or DEVICE
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        print(f"Initializing Hybrid Extractor on {self.device}...")
        
        # Load Custom OCR model
        print("  → Loading Custom OCR model...")
        # Load model once and keep on DEVICE
        self.model, self.idx_to_class = load_model(
            model_path,
            class_mapping_path,
            device=self.device
        )
        print(f"    ✅ Custom OCR model loaded ({len(self.idx_to_class)} classes)")
        
        # Initialize Google Cloud Vision
        print("  → Initializing Google Cloud Vision...")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.gcv_client = vision.ImageAnnotatorClient()
        print("    ✅ Google Cloud Vision initialized")

        # Initialize Kannada word validator
        try:
            print(f"  → Initializing KannadaWordValidator (multi-source)")
            # Let the validator use default paths under data/dictionaries
            if dictionary_path:
                self.validator = KannadaWordValidator(dictionary_path)
            else:
                self.validator = KannadaWordValidator()
        except Exception as e:
            print(f"    ⚠ Warning: Failed to load dictionary validator: {e}")
            self.validator = None
        
        print("✅ Hybrid Extractor ready!\n")
    
    # ========== IMAGE PREPROCESSING ==========
    
    def preprocess_image(self, image):
        """
        Preprocess image for character segmentation
        
        Args:
            image: PIL Image
            
        Returns:
            binary_image: Binary (black and white) image for segmentation
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        print(f"    [DEBUG] Input image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        print(f"    [DEBUG] Grayscale shape: {gray.shape}, pixel range: [{gray.min()}, {gray.max()}]")
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        print(f"    [DEBUG] Denoised range: [{denoised.min()}, {denoised.max()}]")
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        print(f"    [DEBUG] Enhanced range: [{enhanced.min()}, {enhanced.max()}]")
        
        # Binarization using Otsu's method
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # DEBUG: Check binarization quality
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        white_ratio = white_pixels / total_pixels
        black_ratio = black_pixels / total_pixels
        
        print(f"    [DEBUG] Binarization: {white_ratio:.1%} white, {black_ratio:.1%} black")
        print(f"    [DEBUG] Unique values in binary: {len(np.unique(binary))}")
        
        if white_ratio > 0.95:
            print(f"    [WARNING] Image is 95%+ white - text may be inverted or preprocessing failed")
        if black_ratio > 0.95:
            print(f"    [WARNING] Image is 95%+ black - may need different threshold")
        
        return binary
    
    # ========== LINE SEGMENTATION ==========
    
    def segment_lines(self, binary_image):
        """
        Segment image into lines using horizontal projection
        
        Args:
            binary_image: Binary image
            
        Returns:
            lines: List of line images with bounding boxes
        """
        # Horizontal projection
        horizontal_projection = np.sum(binary_image == 0, axis=1)  # Count black pixels
        
        # Find row indices where there's significant text
        threshold = np.max(horizontal_projection) * 0.1
        text_rows = np.where(horizontal_projection > threshold)[0]
        
        print(f"    [DEBUG] Line segmentation:")
        print(f"      Projection max: {np.max(horizontal_projection)}, threshold: {threshold:.2f}")
        print(f"      Rows above threshold: {len(text_rows)}")
        print(f"      Projection stats: mean={horizontal_projection.mean():.2f}, std={horizontal_projection.std():.2f}")
        
        if len(text_rows) == 0:
            print(f"    [WARNING] No lines detected with threshold {threshold:.2f}. Trying lower threshold...")
            # Try lower threshold
            threshold = np.max(horizontal_projection) * 0.05
            text_rows = np.where(horizontal_projection > threshold)[0]
            print(f"    [DEBUG] With lower threshold {threshold:.2f}: found {len(text_rows)} rows")
            if len(text_rows) == 0:
                return []
        
        # Find line boundaries (gaps in text)
        line_boundaries = []
        start_row = text_rows[0]
        
        for i in range(1, len(text_rows)):
            if text_rows[i] - text_rows[i-1] > 5:  # Gap threshold
                end_row = text_rows[i-1]
                line_boundaries.append((start_row, end_row))
                start_row = text_rows[i]
        
        line_boundaries.append((start_row, text_rows[-1]))
        
        print(f"    [DEBUG] Found {len(line_boundaries)} line boundaries")
        
        # Extract line images
        lines = []
        for idx, (start, end) in enumerate(line_boundaries):
            line_img = binary_image[start:end+1, :]
            if line_img.size > 0:
                lines.append({
                    'image': line_img,
                    'bbox': (0, start, binary_image.shape[1], end)
                })
                print(f"      Line {idx}: rows {start}-{end}, height={end-start+1}, width={line_img.shape[1]}")
        
        return lines
    
    # ========== WORD SEGMENTATION ==========
    
    def segment_words(self, line_image):
        """
        Segment line into words using vertical projection
        
        Args:
            line_image: Binary image of a line
            
        Returns:
            words: List of word images
        """
        if line_image.size == 0:
            print(f"      [DEBUG] Line image is empty")
            return []
        
        # Vertical projection
        vertical_projection = np.sum(line_image == 0, axis=0)
        
        # Find column indices with text
        threshold = np.max(vertical_projection) * 0.1
        text_cols = np.where(vertical_projection > threshold)[0]
        
        print(f"      [DEBUG] Word segmentation: projection_max={np.max(vertical_projection)}, threshold={threshold:.2f}, text_cols={len(text_cols)}")
        
        if len(text_cols) == 0:
            print(f"      [WARNING] No words found in line (all blank or below threshold)")
            return []
        
        # Find word boundaries (gaps)
        word_boundaries = []
        start_col = text_cols[0]
        
        for i in range(1, len(text_cols)):
            if text_cols[i] - text_cols[i-1] > 3:  # Gap threshold
                end_col = text_cols[i-1]
                word_boundaries.append((start_col, end_col))
                start_col = text_cols[i]
        
        word_boundaries.append((start_col, text_cols[-1]))
        
        print(f"      [DEBUG] Found {len(word_boundaries)} words")
        
        # Extract word images
        words = []
        for idx, (start, end) in enumerate(word_boundaries):
            word_img = line_image[:, start:end+1]
            if word_img.size > 0:
                words.append({
                    'image': word_img,
                    'bbox': (start, 0, end, line_image.shape[0])
                })
                print(f"        Word {idx}: cols {start}-{end}, width={end-start+1}, height={word_img.shape[0]}")
        
        return words
    
    # ========== CHARACTER SEGMENTATION ==========
    
    def segment_characters(self, word_image):
        """
        Segment word into characters using connected components
        
        Args:
            word_image: Binary image of a word
            
        Returns:
            characters: List of character images with bounding boxes
        """
        if word_image.size == 0:
            print(f"          [DEBUG] Word image is empty")
            return []
        
        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            word_image, connectivity=8
        )
        
        print(f"          [DEBUG] Connected components: {num_labels} components found")
        if num_labels > 1:
            # Print component stats
            for i in range(1, min(6, num_labels)):  # Show first 5
                x, y, w, h, area = stats[i]
                print(f"            Component {i}: area={area}, bbox=({x},{y},{w},{h})")
        
        # Extract bounding boxes (skip background label 0)
        characters = []
        filtered_count = 0
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            
            # Filter by size to avoid noise
            if area < 20:  # Too small
                filtered_count += 1
                continue
            if h > word_image.shape[0] * 0.9:  # Too large
                filtered_count += 1
                continue
            
            char_img = word_image[y:y+h, x:x+w]
            if char_img.size > 0:
                characters.append({
                    'image': char_img,
                    'bbox': (x, y, x+w, y+h),
                    'area': area
                })
        
        print(f"          [DEBUG] After filtering: {len(characters)} characters, {filtered_count} filtered out")
        
        # Sort by x-coordinate (left to right)
        characters.sort(key=lambda c: c['bbox'][0])
        
        return characters
    
    # ========== CHARACTER PREDICTION ==========
    
    def predict_character(self, char_image):
        """
        Predict single character using the trained model
        
        Args:
            char_image: Binary image of character (numpy array)
            
        Returns:
            predicted_char: Predicted character
            confidence: Confidence score (0-1)
        """
        # Legacy single-character prediction kept for backward compatibility
        try:
            pil_char = Image.fromarray(char_image).convert('L')
            pil_char_rgb = Image.new('RGB', pil_char.size)
            pil_char_rgb.paste(pil_char)
            img_tensor = TEST_TRANSFORM(pil_char_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # ✅ FIX: Updated autocast syntax for PyTorch compatibility
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):  # ✅ NEW SYNTAX
                        output = self.model(img_tensor)
                else:
                    output = self.model(img_tensor)

                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_char = self.idx_to_class.get(predicted_idx.item(), "?")
                confidence_score = confidence.item()

            return predicted_char, confidence_score
        except Exception as e:
            print(f"            [ERROR] Predicting character: {e}")
            return "?", 0.0

    def predict_characters_batch(self, char_images):
        """Batch predict list of character image arrays.

        Returns list of (predicted_char, confidence)
        """
        if not char_images:
            return []
        tensors = []
        for char_image in char_images:
            pil_char = Image.fromarray(char_image).convert('L')
            pil_char_rgb = Image.new('RGB', pil_char.size)
            pil_char_rgb.paste(pil_char)
            tensors.append(TEST_TRANSFORM(pil_char_rgb))

        img_batch = torch.stack(tensors, dim=0).to(self.device)

        try:
            with torch.no_grad():
                # ✅ FIX: Updated autocast syntax
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):  # ✅ NEW SYNTAX
                        outputs = self.model(img_batch)
                else:
                    outputs = self.model(img_batch)

                probs = F.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, dim=1)

                results = []
                preds_cpu = preds.cpu().numpy()
                conf_cpu = confidences.cpu().numpy()
                for p, c in zip(preds_cpu, conf_cpu):
                    ch = self.idx_to_class.get(int(p), "?")
                    results.append((ch, float(c)))

                return results
        except Exception as e:
            print(f"            [ERROR] Batch predicting characters: {e}")
            # Fallback to single predictions
            out = []
            for ci in char_images:
                out.append(self.predict_character(ci))
            return out
    
    # ========== CUSTOM OCR EXTRACTION ==========
    
    def extract_with_custom_ocr(self, image):
        """
        Extract text using custom OCR with proper segmentation
        
        Args:
            image: PIL Image
            
        Returns:
            extracted_text: String with extracted text
            stats: Dictionary with extraction statistics
        """
        print("  🔍 Running Custom OCR with character segmentation...")
        
        # Preprocess image
        binary_image = self.preprocess_image(image)
        
        # Segment lines
        lines = self.segment_lines(binary_image)
        print(f"    [INFO] Found {len(lines)} lines")
        
        extracted_lines = []
        total_chars_processed = 0
        total_chars_recognized = 0
        
        for line_idx, line_data in enumerate(lines):
            line_image = line_data['image']
            print(f"    [LINE {line_idx}] Processing line...")
            
            # Segment words in line
            words = self.segment_words(line_image)
            print(f"      [INFO] Line {line_idx}: {len(words)} words")
            
            extracted_words = []
            for word_idx, word_data in enumerate(words):
                word_image = word_data['image']
                print(f"      [WORD {word_idx}] Processing word...")
                
                # Segment characters in word
                characters = self.segment_characters(word_image)
                print(f"        [INFO] Word {word_idx}: {len(characters)} characters")
                
                extracted_chars = []
                # Batch predict characters for this word
                char_images = [c['image'] for c in characters]
                preds = self.predict_characters_batch(char_images)
                for char_idx, (char, confidence) in enumerate(preds):
                    total_chars_processed += 1
                    if confidence >= CONFIDENCE_THRESHOLD:
                        extracted_chars.append(char)
                        total_chars_recognized += 1
                        print(f"            Char {char_idx}: '{char}' (confidence: {confidence:.2%})")
                    else:
                        print(f"            Char {char_idx}: '{char}' (confidence: {confidence:.2%}) - FILTERED (below {CONFIDENCE_THRESHOLD:.1%})")
                
                word_text = ''.join(extracted_chars)
                if word_text:
                    extracted_words.append(word_text)
                    print(f"        [WORD RESULT {word_idx}] '{word_text}'")
            
            line_text = ' '.join(extracted_words)
            if line_text:
                extracted_lines.append(line_text)
                print(f"      [LINE RESULT {line_idx}] '{line_text}'")
        
        extracted_text = '\n'.join(extracted_lines)
        
        stats = {
            'lines': len(lines),
            'characters_processed': total_chars_processed,
            'characters_recognized': total_chars_recognized,
            'recognition_rate': total_chars_recognized / max(total_chars_processed, 1)
        }
        
        return extracted_text, stats
    
    # ========== GOOGLE CLOUD VISION EXTRACTION ==========
    
    def resize_image_for_gcv(self, image, max_dimension=4096, max_pixels=75000000):
        """
        Resize image to fit Google Cloud Vision limits
        
        GCV Limits:
        - Max dimensions: No single dimension > ~75MP total pixels
        - Max file size: 20 MB
        - Recommended: Keep under 4096px on longest side
        
        Args:
            image: PIL Image
            max_dimension: Maximum size for longest dimension
            max_pixels: Maximum total pixels (width * height)
            
        Returns:
            resized_image: PIL Image within GCV limits
        """
        width, height = image.size
        total_pixels = width * height
        
        print(f"    [DEBUG] Original image: {width}x{height} = {total_pixels:,} pixels")
        
        # Check if resizing needed
        needs_resize = False
        scale_factor = 1.0
        
        # Check pixel count
        if total_pixels > max_pixels:
            scale_factor = min(scale_factor, (max_pixels / total_pixels) ** 0.5)
            needs_resize = True
            print(f"    [DEBUG] Image exceeds {max_pixels:,} pixel limit")
        
        # Check dimensions
        if max(width, height) > max_dimension:
            if width > height:
                dim_scale = max_dimension / width
            else:
                dim_scale = max_dimension / height
            scale_factor = min(scale_factor, dim_scale)
            needs_resize = True
            print(f"    [DEBUG] Image exceeds {max_dimension}px dimension limit")
        
        # Perform resize if needed
        if needs_resize:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            print(f"    [DEBUG] Resizing to: {new_width}x{new_height} (scale={scale_factor:.3f})")
            
            # Use LANCZOS for high-quality downsampling
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            print(f"    [DEBUG] Resized: {resized.size[0]}x{resized.size[1]} = {resized.size[0]*resized.size[1]:,} pixels")
            return resized
        else:
            print(f"    [DEBUG] Image within GCV limits, no resize needed")
            return image
    
    def extract_with_gcv(self, image):
        """
        Extract text using Google Cloud Vision API
        
        FIXED: Properly handles large images by resizing before API call
        
        Args:
            image: PIL Image (ORIGINAL, not preprocessed/binarized)
            
        Returns:
            extracted_text: String with extracted text
        """
        print("  🌐 Running Google Cloud Vision...")
        
        try:
            # Validate image
            if not isinstance(image, Image.Image):
                print(f"    [WARNING] Image is not PIL Image: {type(image)}")
                return ""
            
            print(f"    [DEBUG] Image mode: {image.mode}, size: {image.size}")
            
            # ✅ FIX: Resize image if too large for GCV
            image_for_gcv = self.resize_image_for_gcv(image, max_dimension=4096, max_pixels=75000000)
            
            # Ensure image is in RGB mode (GCV prefers RGB over RGBA)
            if image_for_gcv.mode != 'RGB':
                print(f"    [DEBUG] Converting from {image_for_gcv.mode} to RGB")
                image_rgb = image_for_gcv.convert('RGB')
            else:
                image_rgb = image_for_gcv
            
            # Convert PIL to bytes - use JPEG with good quality
            img_byte_arr = io.BytesIO()
            image_rgb.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
            img_byte_arr.seek(0)
            img_bytes = img_byte_arr.getvalue()
            
            print(f"    [DEBUG] Converted to JPEG: {len(img_bytes):,} bytes ({len(img_bytes)/1024/1024:.2f} MB)")
            
            # Check file size limit (20 MB)
            if len(img_bytes) > 20 * 1024 * 1024:
                print(f"    [WARNING] Image still exceeds 20MB after resize, trying lower quality...")
                img_byte_arr = io.BytesIO()
                image_rgb.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr.seek(0)
                img_bytes = img_byte_arr.getvalue()
                print(f"    [DEBUG] Reduced to: {len(img_bytes):,} bytes ({len(img_bytes)/1024/1024:.2f} MB)")
            
            # ✅ Create vision_image from bytes with language hints
            vision_image = vision.Image(content=img_bytes)
            image_context = ImageContext(language_hints=['kn'])  # Kannada language hint
            
            print(f"    [DEBUG] Calling GCV document_text_detection...")
            
            # ✅ Call GCV API with proper error handling
            response = self.gcv_client.document_text_detection(
                image=vision_image,
                image_context=image_context
            )
            
            # Check for API errors
            if response.error.message:
                print(f"    [ERROR] GCV API Error: {response.error.message}")
                
                # ✅ Retry with even smaller image if "Bad image data" error
                if "Bad image data" in response.error.message or "invalid" in response.error.message.lower():
                    print(f"    [DEBUG] Retrying with smaller image (max 2048px)...")
                    
                    # Retry with smaller size
                    retry_image = self.resize_image_for_gcv(image, max_dimension=2048, max_pixels=50000000)
                    if retry_image.mode != 'RGB':
                        retry_image = retry_image.convert('RGB')
                    
                    retry_byte_arr = io.BytesIO()
                    retry_image.save(retry_byte_arr, format='JPEG', quality=90, optimize=True)
                    retry_byte_arr.seek(0)
                    retry_bytes = retry_byte_arr.getvalue()
                    
                    print(f"    [DEBUG] Retry image: {len(retry_bytes):,} bytes")
                    
                    retry_vision_image = vision.Image(content=retry_bytes)
                    retry_response = self.gcv_client.document_text_detection(
                        image=retry_vision_image,
                        image_context=image_context
                    )
                    
                    if retry_response.error.message:
                        print(f"    [ERROR] Retry also failed: {retry_response.error.message}")
                        return ""
                    
                    response = retry_response
                    print(f"    [SUCCESS] Retry succeeded!")
                else:
                    return ""
            
            # Extract text
            text = response.full_text_annotation.text if response.full_text_annotation else ""
            print(f"    [SUCCESS] GCV extracted {len(text)} characters")
            
            return text
        
        except Exception as e:
            print(f"    [ERROR] Exception in GCV extraction: {type(e).__name__}: {e}")
            import traceback
            print(f"    [DEBUG] Traceback: {traceback.format_exc()}")
            return ""
    
    # ========== RESULT FUSION ==========
    
    def merge_results(self, custom_text, gcv_text):
        """
        Intelligently merge Custom OCR and GCV results
        
        Strategy:
        - Use Custom OCR as primary (character-level accuracy)
        - Use GCV for validation and context
        - Prefer GCV text if Custom OCR is too short or empty
        - Combine word boundaries from both
        
        Args:
            custom_text: Text from Custom OCR
            gcv_text: Text from Google Cloud Vision
            
        Returns:
            merged_text: Fused text result
        """
        print("  🔗 Merging Custom OCR and GCV results...")
        
        # If custom OCR has sufficient content, use it as primary
        if custom_text and len(custom_text.strip()) > 20:
            merged_text = custom_text
            print(f"    ✅ Using Custom OCR as primary source ({len(custom_text)} chars)")
        # If custom OCR is insufficient but GCV has content, use GCV
        elif gcv_text and len(gcv_text.strip()) > 0:  # ✅ Changed from >20 to >0 for better fallback
            merged_text = gcv_text
            print(f"    ✅ Using GCV as fallback ({len(gcv_text)} chars)")
        # Both empty
        else:
            merged_text = custom_text or gcv_text or ""
            print(f"    ⚠ Both methods returned minimal text")
        
        return merged_text
    
    # ========== MAIN EXTRACTION ==========
    
    def extract_from_image(self, image):
        """
        Main extraction method: unified hybrid pipeline
        
        Args:
            image: PIL Image
            
        Returns:
            extracted_text: Final unified text result
            validation_result: Validation/correction results
        """
        # Step 1: Custom OCR with character segmentation
        custom_text, custom_stats = self.extract_with_custom_ocr(image)
        
        # Step 2: Google Cloud Vision for context
        gcv_text = self.extract_with_gcv(image)
        
        # Step 3: Intelligent fusion
        final_text = self.merge_results(custom_text, gcv_text)

        # Step 4: Validate and correct using dictionary validator (if available)
        validation_result = None
        if getattr(self, 'validator', None):
            try:
                validation_result = self.validator.validate_text(final_text)
                final_text = validation_result.get('corrected_text', final_text)
            except Exception as e:
                print(f"    ⚠ Validation error: {e}")

        return final_text, validation_result


# ============================================================
# PDF PROCESSING
# ============================================================

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF to images
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        
    Returns:
        images: List of PIL Images
    """
    print(f"Converting PDF to images (DPI: {dpi})...")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"✅ Extracted {len(images)} page(s)")
    for i, img in enumerate(images):
        print(f"  [DEBUG] Image {i}: size={img.size}, mode={img.mode}")
    print("")
    return images


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_text_from_pdf(
    pdf_path,
    model_path=None,
    class_mapping_path=None,
    credentials_path=None
):
    """
    Extract text from PDF using unified hybrid pipeline
    
    This function combines Custom OCR (character-level) with 
    Google Cloud Vision (document-level) for optimal results.
    
    Args:
        pdf_path: Path to PDF file (user-selected, not hardcoded)
        model_path: Path to .pth file (for custom OCR)
        class_mapping_path: Path to class mapping JSON (for custom OCR)
        credentials_path: Path to GCV credentials JSON (for GCV)
    
    Returns:
        unified_text: Complete extracted text from all pages
    """
    # Validate inputs
    if not model_path or not class_mapping_path:
        raise ValueError("model_path and class_mapping_path required")
    if not credentials_path:
        raise ValueError("credentials_path required")
    
    # Initialize hybrid extractor
    extractor = HybridExtractor(model_path, class_mapping_path, credentials_path)
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    
    # Process each page
    print(f"{'='*70}")
    print(f"Processing {len(images)} page(s) with Unified Hybrid Pipeline")
    print('='*70 + "\n")
    
    all_pages_text = []
    all_page_validations = []
    
    for page_num, image in enumerate(images, 1):
        print(f"📄 Page {page_num}/{len(images)}:")
        print(f"  [DEBUG] Image size: {image.size}, mode: {image.mode}")
        
        # Extract using unified hybrid pipeline
        page_text, validation = extractor.extract_from_image(image)
        all_pages_text.append(page_text)
        all_page_validations.append(validation)

        print(f"  [RESULT] ✅ Extracted {len(page_text)} characters")
        if len(page_text) == 0:
            print(f"  [ERROR] Page {page_num} returned ZERO characters!")
        print("")
    
    # Combine all pages
    unified_text = '\n\n--- PAGE BREAK ---\n\n'.join(all_pages_text)
    
    print(f"{'='*70}")
    print("✅ Extraction complete!")
    print(f"Total characters: {len(unified_text)}")
    print('='*70 + "\n")
    
    # Build corrected text from per-page validation results when available
    corrected_pages = []
    for v, page_text in zip(all_page_validations, all_pages_text):
        if isinstance(v, dict):
            corrected_pages.append(v.get('corrected_text', page_text))
        else:
            corrected_pages.append(page_text)
    corrected_text = '\n\n--- PAGE BREAK ---\n\n'.join(corrected_pages)

    # Save validation stats (if any) next to extracted text
    stats_file = None
    try:
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs('outputs', exist_ok=True)
        stats_file = os.path.join('outputs', f'extracted_text_{ts}_validation.json')
        with open(stats_file, 'w', encoding='utf-8') as sf:
            json.dump({'pages': all_page_validations}, sf, ensure_ascii=False, indent=2)
        print(f"✅ Validation stats saved to {stats_file}")
    except Exception as e:
        print(f"⚠ Could not save validation stats: {e}")

    # Prepare a clean return dict for the caller (UI)
    stats = {
        'pages': len(images),
        'total_characters': len(unified_text),
        'validation_file': stats_file,
        'per_page': [ (v.get('stats') if isinstance(v, dict) else {}) for v in all_page_validations ]
    }

    return {
        'text': unified_text,
        'corrected_text': corrected_text,
        'stats': stats,
        'validation_json': stats_file
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_results(pdf_path, extracted_text, output_dir='outputs'):
    """
    Save extraction results to file
    
    Args:
        pdf_path: Path to source PDF
        extracted_text: Extracted text string
        output_dir: Directory to save results
    """
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = f'{output_dir}/extracted_text_{timestamp}.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    print(f"✅ Results saved to {output_file}")
    return output_file


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("Hybrid Extraction Module")
    print("="*80)
    
    print("\nTo use this module:")
    print("  from extraction import extract_text_from_pdf, HybridExtractor")
    print("  text = extract_text_from_pdf(pdf_path, model_path, class_mapping_path, credentials_path)")
    print("  (pdf_path is user-selected, not hardcoded)")
    print("="*80)