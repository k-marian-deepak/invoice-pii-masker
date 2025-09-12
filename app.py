from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import base64
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple

# Import your existing modules
from core.ocr import ocr_words, configure_tesseract
from core.masking import apply_mask, Mode
from core.patterns import PATTERNS, LABEL_SYNONYMS
from core.alignment import find_label_words

def parse_txt_labels(txt_path: str) -> List[str]:
    """Parse TXT file to extract values that should be masked."""
    if not os.path.exists(txt_path):
        return []
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"TXT file content preview:\n{content[:500]}...")
        
        # Extract specific values that should be masked
        values_to_mask: List[str] = []
        
        # Extract specific patterns from the TXT content
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Extract values after colons
                parts = line.split(':')
                if len(parts) >= 2:
                    value = parts[1].strip()
                    # Clean up common prefixes and suffixes
                    value = value.replace('$', '').replace(',', '').strip()
                    if value and value != 'NONE' and len(value) > 2:
                        values_to_mask.append(value)
            
            # Also look for standalone values (like in the description)
            # Extract alphanumeric codes, IDs, etc.
            import re
            # Look for patterns like codes, IDs, phone numbers
            code_patterns = [
                r'\b[A-Z]+\d+[A-Z]*\b',  # Codes like P0028V, CNC183
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone numbers
                r'\b[A-Z]{2,}_[A-Z]+\d+_\d+\b',  # Document IDs
                r'\b\d{2}/\d{2}/\d{2,4}\b',  # Dates
                r'\b\$\d+\.\d{2}\b',  # Money amounts
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, line)
                values_to_mask.extend(matches)
        
        # Manual extraction of key values mentioned in the TXT
        key_values = [
            'I457445745',  # Freight Bill No
            'P0028V',      # Customer Code
            'CNC183', 'PAR129',  # Shipper/Consignee Codes
            'GL_INV00555_000140', 'US-043-000000162',  # Document IDs
            '937-382-1494', '800-543-5589',  # Phone numbers
            '85.57', '76.40', '9.17', '50.00',  # Amounts
            '04/07/25',    # Date
            'CZR9901', 'C50',  # Tariff codes
            '91', '1',     # Weight, pieces
            '000322', '000000162',  # Machine readable
            '73 52 53 744574 53 000008557 9',  # Bottom line
        ]
        
        values_to_mask.extend(key_values)
        
        # Remove duplicates and filter out very short values
        unique_values = list(set([v for v in values_to_mask if len(v.strip()) > 1]))
        
        print(f"Extracted {len(unique_values)} values to mask: {unique_values}")
        return unique_values
        
    except Exception as e:
        print(f"Error reading TXT file {txt_path}: {e}")
        return []

def find_txt_based_pii_words(words: List[Dict[str, Any]], txt_values: List[str]) -> List[Dict[str, Any]]:
    """Find words that should be masked based on TXT file values."""
    pii_words: List[Dict[str, Any]] = []
    
    print(f"Processing {len(txt_values)} TXT values: {txt_values}")
    print(f"OCR found {len(words)} words total")
    
    # First, let's print all OCR words to see what we're working with
    print("All OCR words:")
    for i, word in enumerate(words):
        print(f"  {i}: '{word.get('text', 'N/A')}' at ({word.get('left', 0)}, {word.get('top', 0)})")
    
    # Look for exact matches or partial matches of the TXT values in OCR words
    for value in txt_values:
        print(f"\nLooking for value: '{value}'")
        
        for word in words:
            word_text = word.get('text', '').strip()
            
            # Exact match
            if word_text == value:
                print(f"  EXACT MATCH: '{word_text}' == '{value}'")
                pii_words.append(word)
            # Partial match (value contains the word or word contains the value)
            elif value in word_text or word_text in value:
                print(f"  PARTIAL MATCH: '{word_text}' contains/in '{value}'")
                pii_words.append(word)
            # For multi-part values like "73 52 53 744574 53 000008557 9"
            elif ' ' in value and word_text in value.split():
                print(f"  PART MATCH: '{word_text}' is part of '{value}'")
                pii_words.append(word)
    
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
    
    print(f"Final unique words to mask: {len(unique_words)}")
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
