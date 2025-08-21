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
            
            # Detect patterns in OCR text
            patterns = detect_patterns(words)
            
            # Find PII words to mask
            pii_words = find_pii_words(words)
            
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
