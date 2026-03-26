from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import base64
import re
from difflib import SequenceMatcher
from collections import defaultdict
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple

# Import your existing modules
from core.ocr import ocr_words, configure_tesseract
from core.masking import apply_mask, Mode
from core.patterns import PATTERNS, LABEL_SYNONYMS
from core.alignment import find_label_words

def parse_txt_labels(txt_path: str) -> List[str]:
    """Parse TXT file and extract likely sensitive values to mask."""
    if not os.path.exists(txt_path):
        return []

    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        candidates: List[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if ':' in line:
                _, rhs = line.split(':', 1)
                rhs = rhs.strip()
                if rhs:
                    candidates.append(rhs)

            candidates.extend(re.findall(r'[A-Z]{2,}(?:[-_][A-Z0-9]+)+', line))
            candidates.extend(re.findall(r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', line))
            candidates.extend(re.findall(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', line, flags=re.IGNORECASE))
            candidates.extend(re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b', line))
            candidates.extend(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line))
            candidates.extend(re.findall(r'\b[A-Z0-9]{2,}[/-][A-Z0-9-]{2,}\b', line))
            candidates.extend(re.findall(r'\b\d{6,}\b', line))

        cleaned: List[str] = []
        seen: set[str] = set()
        for value in candidates:
            token = value.strip(" .,;:()[]{}\t\n\r")
            if len(token) < 3:
                continue
            if token.lower().startswith(("highly sensitive", "transaction", "financial", "internal")):
                continue
            key = token.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(token)

        print(f"Extracted {len(cleaned)} values from TXT file")
        return cleaned

    except Exception as e:
        print(f"Error reading TXT file {txt_path}: {e}")
        return []

def normalize_token(text: str) -> str:
    """Normalize text for OCR-tolerant comparisons."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def similarity(a: str, b: str) -> float:
    """String similarity in [0,1]."""
    return SequenceMatcher(None, a, b).ratio()

def find_txt_based_pii_words(words: List[Dict[str, Any]], txt_values: List[str]) -> List[Dict[str, Any]]:
    """Find words that should be masked based on TXT file values."""
    pii_words: List[Dict[str, Any]] = []

    print(f"Processing {len(txt_values)} TXT values")
    if not txt_values or not words:
        return pii_words

    words_by_line: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    normalized_to_words: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for word in words:
        line_num = int(word.get("line_num", 0))
        words_by_line[line_num].append(word)

        word_norm = normalize_token(str(word.get("text", "")))
        if word_norm:
            normalized_to_words[word_norm].append(word)

    for line_num in words_by_line:
        words_by_line[line_num].sort(key=lambda item: int(item.get("left", 0)))

    for value in txt_values:
        tokens = [normalize_token(t) for t in re.split(r'\s+', value) if normalize_token(t)]
        if not tokens:
            continue

        if len(tokens) == 1:
            target = tokens[0]
            if target in normalized_to_words:
                pii_words.extend(normalized_to_words[target])
                continue

            if len(target) >= 6:
                for word_norm, grouped_words in normalized_to_words.items():
                    if len(word_norm) < 6:
                        continue
                    if similarity(target, word_norm) >= 0.88:
                        pii_words.extend(grouped_words)
            continue

        joined_target = ''.join(tokens)
        for line_words in words_by_line.values():
            norm_line = [normalize_token(str(w.get("text", ""))) for w in line_words]
            window_size = len(tokens)
            for idx in range(0, len(norm_line) - window_size + 1):
                window = norm_line[idx:idx + window_size]
                if window == tokens:
                    pii_words.extend(line_words[idx:idx + window_size])
                    continue

                joined_window = ''.join(window)
                if len(joined_target) >= 8 and len(joined_window) >= 8 and similarity(joined_target, joined_window) >= 0.86:
                    pii_words.extend(line_words[idx:idx + window_size])

    unique_words: List[Dict[str, Any]] = []
    seen_positions: set[Tuple[int, int, int, int]] = set()
    for word in pii_words:
        key = (
            int(word.get('left', 0)),
            int(word.get('top', 0)),
            int(word.get('width', 0)),
            int(word.get('height', 0)),
        )
        if key in seen_positions:
            continue
        seen_positions.add(key)
        unique_words.append(word)

    print(f"Final unique words to mask from TXT matching: {len(unique_words)}")
    return unique_words

def detect_patterns(words: List[Dict[str, Any]]) -> List[str]:
    """Detect patterns in OCR words and return matching pattern types."""
    detected: List[str] = []
    for word_info in words:
        text = word_info.get('text', '')
        for pattern_regex in PATTERNS.values():
            if pattern_regex.search(text):
                detected.append(text)  # Store the actual detected text
    return list(set(detected))  # Remove duplicates

def find_pii_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find words that contain PII patterns or are related to label synonyms."""
    pii_words: List[Dict[str, Any]] = []
    
    # Find words matching direct patterns
    for word_info in words:
        text = word_info.get('text', '')
        for pattern_regex in PATTERNS.values():
            if pattern_regex.search(text):
                pii_words.append(word_info)
                break
    
    # Find words that are near labels
    for synonyms in LABEL_SYNONYMS.values():
        label_words = find_label_words(words, synonyms)
        pii_words.extend(label_words)
    
    # Remove duplicates based on word position
    unique_words: List[Dict[str, Any]] = []
    for word in pii_words:
        is_duplicate = False
        for existing in unique_words:
            if (word.get('left') == existing.get('left') and 
                word.get('top') == existing.get('top')):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_words.append(word)
    
    return unique_words

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    return (0, 0, 0)  # Default to black

def word_to_box(word: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Convert word dict to bounding box tuple."""
    return (
        word.get('left', 0),
        word.get('top', 0), 
        word.get('width', 0),
        word.get('height', 0)
    )

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Define upload folder
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create temp upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/mask', methods=['POST'])
def mask_image():
    """Process and mask the uploaded image."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Get parameters
        mode_str = request.form.get('mode', 'black')
        color_hex = request.form.get('color', '#000000')
        pixel_size = int(request.form.get('pixel_size', 12))
        tesseract_cmd = request.form.get('tesseract_cmd')
        
        # Validate mode
        mode: Mode = 'black'  # Default
        if mode_str in ['black', 'blur', 'pixelate', 'color']:
            mode = mode_str  # type: ignore
        
        # Convert color
        color_rgb = hex_to_rgb(color_hex)
        
        # Configure Tesseract if path provided
        if tesseract_cmd:
            configure_tesseract(tesseract_cmd)
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Load and process image
            image = Image.open(filepath)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for masking
            image_array = np.array(image)
            
            # Perform OCR
            words = ocr_words(image)  # Pass PIL Image, not numpy array
            print(f"OCR extracted {len(words)} words from image")
            
            # Debug: Print first few words
            if words:
                print("First few OCR words:")
                for i, word in enumerate(words[:5]):
                    print(f"  {i}: {word}")
            
            # Try to find corresponding TXT file for this image
            base_filename = os.path.splitext(filename)[0]  # Remove extension
            txt_filename = base_filename + '.txt'
            
            # Look for TXT file in dataset/texts or data/texts directories
            txt_paths = [
                os.path.join('dataset', 'texts', txt_filename),
                os.path.join('data', 'texts', txt_filename),
                os.path.join('data', txt_filename)
            ]
            
            txt_values = []
            txt_found = False
            for txt_path in txt_paths:
                if os.path.exists(txt_path):
                    txt_values = parse_txt_labels(txt_path)
                    txt_found = True
                    print(f"Found TXT file: {txt_path} with {len(txt_values)} values")
                    print(f"TXT values: {txt_values}")
                    break
            
            if not txt_found:
                print(f"No TXT file found for {base_filename}. Checked paths: {txt_paths}")
            
            # Find PII words to mask based on TXT file or fallback to pattern matching
            if txt_found and txt_values:
                # Use TXT-based masking for better accuracy
                pii_words = find_txt_based_pii_words(words, txt_values)
                print(f"Using TXT-based masking with {len(txt_values)} values, found {len(pii_words)} words to mask")
            else:
                # Fallback to pattern-based masking
                pii_words = find_pii_words(words)
                print(f"Using pattern-based masking, found {len(pii_words)} words to mask")
            
            print(f"Final PII words to mask: {len(pii_words)}")
            if pii_words:
                print("Words to be masked:")
                for i, word in enumerate(pii_words):
                    print(f"  {i}: {word.get('text', 'N/A')} at ({word.get('left', 0)}, {word.get('top', 0)})")
            
            # Detect patterns in OCR text for statistics
            patterns = detect_patterns(words)
            
            # Create a copy for masking
            masked_array = image_array.copy()
            
            # Apply masking to each PII word
            for word in pii_words:
                box = word_to_box(word)
                apply_mask(masked_array, box, mode=mode, color=color_rgb, pixel_size=pixel_size)
            
            # Convert back to PIL Image
            masked_image = Image.fromarray(masked_array)
            
            # Convert images to base64 for JSON response
            original_buffer = io.BytesIO()
            image.save(original_buffer, format='PNG')
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
            
            masked_buffer = io.BytesIO()
            masked_image.save(masked_buffer, format='PNG')
            masked_base64 = base64.b64encode(masked_buffer.getvalue()).decode()
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'original_image': f'data:image/png;base64,{original_base64}',
                'masked_image': f'data:image/png;base64,{masked_base64}',
                'detected_patterns': len(patterns),
                'masked_items': len(pii_words)
            })
            
        except Exception as processing_error:
            # Clean up temporary file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise processing_error
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/mask_with_txt', methods=['POST'])
def mask_image_with_txt():
    """Process and mask the uploaded image using a specific TXT file."""
    try:
        # Check if both files were uploaded
        if 'image_file' not in request.files or 'txt_file' not in request.files:
            return jsonify({'error': 'Both image and TXT files are required'}), 400
        
        image_file = request.files['image_file']
        txt_file = request.files['txt_file']
        
        if (image_file.filename == '' or image_file.filename is None or 
            txt_file.filename == '' or txt_file.filename is None):
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        if not txt_file.filename.endswith('.txt'):
            return jsonify({'error': 'TXT file must have .txt extension'}), 400
        
        # Get parameters
        mode_str = request.form.get('mode', 'black')
        color_hex = request.form.get('color', '#000000')
        pixel_size = int(request.form.get('pixel_size', 12))
        tesseract_cmd = request.form.get('tesseract_cmd')
        
        # Validate mode
        mode: Mode = 'black'
        if mode_str in ['black', 'blur', 'pixelate', 'color']:
            mode = mode_str  # type: ignore
        
        # Convert color
        color_rgb = hex_to_rgb(color_hex)
        
        # Configure Tesseract if path provided
        if tesseract_cmd:
            configure_tesseract(tesseract_cmd)
        
        # Save files temporarily
        image_filename = secure_filename(image_file.filename)
        txt_filename = secure_filename(txt_file.filename)
        
        image_filepath = os.path.join(UPLOAD_FOLDER, image_filename)
        txt_filepath = os.path.join(UPLOAD_FOLDER, txt_filename)
        
        image_file.save(image_filepath)
        txt_file.save(txt_filepath)
        
        try:
            # Load and process image
            image = Image.open(image_filepath)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for masking
            image_array = np.array(image)
            
            # Perform OCR
            words = ocr_words(image)
            
            # Parse TXT file for masking labels
            txt_labels = parse_txt_labels(txt_filepath)
            print(f"Loaded {len(txt_labels)} labels from TXT file: {txt_labels}")
            
            # Use TXT-based masking
            pii_words = find_txt_based_pii_words(words, txt_labels)
            print(f"Found {len(pii_words)} words to mask based on TXT labels")
            
            # Create a copy for masking
            masked_array = image_array.copy()
            
            # Apply masking to each PII word
            for word in pii_words:
                box = word_to_box(word)
                apply_mask(masked_array, box, mode=mode, color=color_rgb, pixel_size=pixel_size)
            
            # Convert back to PIL Image
            masked_image = Image.fromarray(masked_array)
            
            # Convert images to base64 for JSON response
            original_buffer = io.BytesIO()
            image.save(original_buffer, format='PNG')
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
            
            masked_buffer = io.BytesIO()
            masked_image.save(masked_buffer, format='PNG')
            masked_base64 = base64.b64encode(masked_buffer.getvalue()).decode()
            
            # Clean up temporary files
            os.remove(image_filepath)
            os.remove(txt_filepath)
            
            return jsonify({
                'success': True,
                'original_image': f'data:image/png;base64,{original_base64}',
                'masked_image': f'data:image/png;base64,{masked_base64}',
                'txt_labels_used': txt_labels,
                'masked_items': len(pii_words)
            })
            
        except Exception as processing_error:
            # Clean up temporary files on error
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            if os.path.exists(txt_filepath):
                os.remove(txt_filepath)
            raise processing_error
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
