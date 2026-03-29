# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import base64
import json
import re
from difflib import SequenceMatcher
from collections import defaultdict
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, cast

# Import your existing modules
from core.ocr import ocr_words, configure_tesseract
from core.masking import apply_mask, Mode
from core.patterns import PATTERNS, LABEL_SYNONYMS
from core.alignment import find_label_words, right_of, below, box_of_word

SENSITIVE_KEYWORDS = {
    "account", "routing", "swift", "tax", "duns", "invoice", "bill", "shipper",
    "consignee", "customer", "phone", "email", "charges", "discount", "total",
    "date", "weight", "code", "id", "number", "remit", "sn", "tracking",
}

MIN_CONFIDENCE = 20.0
MEMORY_PATH = os.path.join("data", "memory.json")
MIN_MEMORY_TOKEN_COUNT = 2


def _word_confidence(word: Dict[str, Any]) -> float:
    raw = word.get("conf", -1.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return -1.0


def _filter_confident_words(words: List[Dict[str, Any]], min_conf: float = MIN_CONFIDENCE) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for word in words:
        conf = _word_confidence(word)
        # Keep unknown confidence values (-1) to avoid dropping valid OCR tokens.
        if conf == -1.0 or conf >= min_conf:
            filtered.append(word)
    return filtered


def _dedupe_words_by_box(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_words: List[Dict[str, Any]] = []
    seen_positions: set[Tuple[int, int, int, int]] = set()
    for word in words:
        key = (
            int(word.get("left", 0)),
            int(word.get("top", 0)),
            int(word.get("width", 0)),
            int(word.get("height", 0)),
        )
        if key in seen_positions:
            continue
        seen_positions.add(key)
        unique_words.append(word)
    return unique_words


def _group_words_by_line(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    words_by_line: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for word in words:
        line_key = (
            int(word.get("block_num", 0)),
            int(word.get("par_num", 0)),
            int(word.get("line_num", 0)),
        )
        words_by_line[line_key].append(word)

    lines: List[List[Dict[str, Any]]] = []
    for line_words in words_by_line.values():
        lines.append(sorted(line_words, key=lambda item: int(item.get("left", 0))))
    return lines


def _line_string_with_spans(line_words: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for i, word in enumerate(line_words):
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor))
    return "".join(parts), spans


def _term_tokens(term: str) -> List[str]:
    return [normalize_token(t) for t in term.split() if normalize_token(t)]


def _load_runtime_label_synonyms() -> Dict[str, List[str]]:
    data = _load_runtime_memory()
    synonyms: Dict[str, List[str]] = {k: list(v) for k, v in LABEL_SYNONYMS.items()}
    mem_synonyms_raw = data.get("label_synonyms", {})
    if isinstance(mem_synonyms_raw, dict):
        mem_synonyms = cast(Dict[str, Any], mem_synonyms_raw)
        for key, values_raw in mem_synonyms.items():
            current = set(synonyms.get(key, []))
            if isinstance(values_raw, list):
                values_list = cast(List[Any], values_raw)
                for value in values_list:
                    if isinstance(value, str) and value.strip():
                        current.add(value.strip())
            if key.strip():
                current.add(key.strip())
            synonyms[key] = sorted(current)
    return synonyms


def _load_runtime_memory() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_PATH):
        return {"label_synonyms": LABEL_SYNONYMS.copy(), "vendors": {}, "value_tokens": {}}

    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return cast(Dict[str, Any], data)
    except Exception as e:
        print(f"Warning: failed to load memory from {MEMORY_PATH}: {e}")

    return {"label_synonyms": LABEL_SYNONYMS.copy(), "vendors": {}, "value_tokens": {}}


def _save_runtime_memory(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save memory to {MEMORY_PATH}: {e}")


def _should_remember_token(text: str) -> bool:
    normalized = normalize_token(text)
    if len(normalized) < 5:
        return False

    has_digit = any(ch.isdigit() for ch in normalized)
    has_mixed = any(ch.isdigit() for ch in normalized) and any(ch.isalpha() for ch in normalized)
    is_email_like = "@" in text
    is_money_like = bool(re.search(r"\d[\d,]*\.\d{2}", text))
    return has_digit or has_mixed or is_email_like or is_money_like


def _remember_successful_matches(txt_values: List[str], pii_words: List[Dict[str, Any]]) -> None:
    if not txt_values or not pii_words:
        return

    memory = _load_runtime_memory()
    value_tokens_raw = memory.setdefault("value_tokens", {})
    if not isinstance(value_tokens_raw, dict):
        value_tokens_raw = {}
        memory["value_tokens"] = value_tokens_raw

    value_tokens = cast(Dict[str, Any], value_tokens_raw)
    for word in pii_words:
        token = str(word.get("text", "")).strip()
        normalized = normalize_token(token)
        if not normalized or not _should_remember_token(token):
            continue

        entry_raw = value_tokens.get(normalized)
        if isinstance(entry_raw, dict):
            entry = cast(Dict[str, Any], entry_raw)
        else:
            entry = {"count": 0, "samples": []}

        count = int(entry.get("count", 0)) + 1
        samples_raw = entry.get("samples", [])
        samples = cast(List[str], samples_raw) if isinstance(samples_raw, list) else []
        if token not in samples and len(samples) < 5:
            samples.append(token)

        entry["count"] = count
        entry["samples"] = samples
        value_tokens[normalized] = entry

    _save_runtime_memory(memory)


def _get_memory_value_tokens() -> Dict[str, int]:
    memory = _load_runtime_memory()
    raw = memory.get("value_tokens", {})
    if not isinstance(raw, dict):
        return {}

    memory_tokens: Dict[str, int] = {}
    entries = cast(Dict[str, Any], raw)
    for token, entry_raw in entries.items():
        if len(token) < 5:
            continue
        if isinstance(entry_raw, dict):
            entry = cast(Dict[str, Any], entry_raw)
            count = int(entry.get("count", 0))
        else:
            count = 0
        if count >= MIN_MEMORY_TOKEN_COUNT:
            memory_tokens[token] = count
    return memory_tokens

def _is_noise_candidate(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return True
    if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
        return True
    if len(lower.split()) > 14 and not re.search(r'\d|\$|@', lower):
        return True
    if lower.startswith((
        "highly sensitive", "transaction", "financial", "internal", "masking these",
        "these are unique", "this includes", "this is commercially",
    )):
        return True
    return False

def _looks_sensitive(text: str) -> bool:
    token = text.strip()
    if len(token) < 3:
        return False
    if _is_noise_candidate(token):
        return False

    lower = token.lower()
    words = token.split()

    if re.search(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', token, flags=re.IGNORECASE):
        return True
    if re.search(r'\b\d{3}[- )]?\d{3}[- ]?\d{4}\b', token):
        return True
    if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', token):
        return True
    if re.search(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\b', token):
        return True
    if re.search(r'\b\d{6,}\b', token):
        return True
    if re.search(r'\b[A-Z0-9]{2,}[/-][A-Z0-9-]{2,}\b', token):
        return True
    if any(k in lower for k in SENSITIVE_KEYWORDS):
        return True

    uppercase_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(words) <= 8 and len(uppercase_words) >= 2:
        return True

    return False

def _extract_line_candidates(line: str) -> List[str]:
    candidates: List[str] = []

    if ':' in line:
        _, rhs = line.split(':', 1)
        rhs = rhs.strip()
        if rhs:
            candidates.append(rhs)
            for part in re.split(r'\s+and\s+|[;,]|\s+/\s+', rhs, flags=re.IGNORECASE):
                part = part.strip()
                if part:
                    candidates.append(part)

    candidates.extend(re.findall(r'[A-Z]{2,}(?:[-_][A-Z0-9]+)+', line))
    candidates.extend(re.findall(r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', line))
    candidates.extend(re.findall(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', line, flags=re.IGNORECASE))
    candidates.extend(re.findall(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\b', line))
    candidates.extend(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', line))
    candidates.extend(re.findall(r'\b[A-Z0-9]{2,}[/-][A-Z0-9-]{2,}\b', line))
    candidates.extend(re.findall(r'\b\d{6,}\b', line))

    return candidates

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
            candidates.extend(_extract_line_candidates(line))

        cleaned: List[str] = []
        seen: set[str] = set()
        for value in candidates:
            token = value.strip(" .,;:()[]{}\t\n\r")
            if not _looks_sensitive(token):
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
    words = _filter_confident_words(words)

    print(f"Processing {len(txt_values)} TXT values")
    if not txt_values or not words:
        return pii_words

    words_by_line: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    normalized_to_words: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for word in words:
        line_key = (
            int(word.get("block_num", 0)),
            int(word.get("par_num", 0)),
            int(word.get("line_num", 0)),
        )
        words_by_line[line_key].append(word)

        word_norm = normalize_token(str(word.get("text", "")))
        if word_norm:
            normalized_to_words[word_norm].append(word)

    for line_num in words_by_line:
        words_by_line[line_num].sort(key=lambda item: int(item.get("left", 0)))

    for value in txt_values:
        tokens = [normalize_token(t) for t in re.split(r'\s+', value) if normalize_token(t)]
        if not tokens:
            continue

        has_digit_signal = any(any(ch.isdigit() for ch in t) for t in tokens)
        alpha_only_multi = len(tokens) > 1 and all(t.isalpha() for t in tokens)

        if len(tokens) > 10 and not has_digit_signal:
            continue

        if len(tokens) == 1:
            target = tokens[0]
            if target in normalized_to_words:
                pii_words.extend(normalized_to_words[target])
                continue

            if len(target) >= 6:
                fuzzy_threshold = 0.84 if target.isdigit() else 0.86
                for word_norm, grouped_words in normalized_to_words.items():
                    if len(word_norm) < 6:
                        continue
                    if similarity(target, word_norm) >= fuzzy_threshold:
                        pii_words.extend(grouped_words)
            continue

        if alpha_only_multi and len(tokens) > 4:
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
                if (
                    has_digit_signal
                    and len(joined_target) >= 8
                    and len(joined_window) >= 8
                    and similarity(joined_target, joined_window) >= 0.86
                ):
                    pii_words.extend(line_words[idx:idx + window_size])

    unique_words = _dedupe_words_by_box(pii_words)

    print(f"Final unique words to mask from TXT matching: {len(unique_words)}")
    return unique_words

def detect_patterns(words: List[Dict[str, Any]]) -> List[str]:
    """Detect patterns in OCR words and return matching pattern types."""
    detected: List[str] = []
    for word_info in _find_pattern_words(words):
        detected.append(str(word_info.get("text", "")))
    return sorted(set(detected))


def _find_pattern_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    # Word-level detection (works well when OCR keeps tokens intact).
    for word_info in words:
        text = str(word_info.get("text", ""))
        for pattern_regex in PATTERNS.values():
            if pattern_regex.search(text):
                matches.append(word_info)
                break

    # Line-level detection (captures split tokens like GSTIN/amount chunks).
    for line_words in _group_words_by_line(words):
        line_text, spans = _line_string_with_spans(line_words)
        if not line_text or len(spans) != len(line_words):
            continue
        for pattern_regex in PATTERNS.values():
            for match in pattern_regex.finditer(line_text):
                m_start, m_end = match.span()
                for idx, (w_start, w_end) in enumerate(spans):
                    if w_start < m_end and w_end > m_start:
                        matches.append(line_words[idx])

    return _dedupe_words_by_box(matches)


def _find_label_anchor_boxes(words: List[Dict[str, Any]], label_terms: List[str]) -> List[Tuple[int, int, int, int]]:
    anchors: List[Tuple[int, int, int, int]] = []

    # Keep the existing exact single-token anchor behavior.
    single_terms = [t for t in label_terms if len(t.split()) == 1]
    if single_terms:
        for hit in find_label_words(words, single_terms):
            anchors.append(box_of_word(hit))

    # Add multi-token label matching across each OCR line.
    term_tokens = [_term_tokens(term) for term in label_terms]
    multi_term_tokens = [tokens for tokens in term_tokens if len(tokens) > 1]
    if not multi_term_tokens:
        return anchors

    for line_words in _group_words_by_line(words):
        norm_line = [normalize_token(str(w.get("text", ""))) for w in line_words]
        for tokens in multi_term_tokens:
            width = len(tokens)
            for idx in range(0, len(norm_line) - width + 1):
                if norm_line[idx:idx + width] != tokens:
                    continue
                matched_words = line_words[idx:idx + width]
                x1 = min(int(w.get("left", 0)) for w in matched_words)
                y1 = min(int(w.get("top", 0)) for w in matched_words)
                x2 = max(int(w.get("left", 0)) + int(w.get("width", 0)) for w in matched_words)
                y2 = max(int(w.get("top", 0)) + int(w.get("height", 0)) for w in matched_words)
                anchors.append((x1, y1, x2 - x1, y2 - y1))

    seen: set[Tuple[int, int, int, int]] = set()
    unique_anchors: List[Tuple[int, int, int, int]] = []
    for box in anchors:
        if box in seen:
            continue
        seen.add(box)
        unique_anchors.append(box)
    return unique_anchors


def _looks_like_value_token(text: str) -> bool:
    token = text.strip()
    if not token:
        return False
    if any(regex.search(token) for regex in PATTERNS.values()):
        return True

    normalized = normalize_token(token)
    if not normalized:
        return False
    if len(normalized) >= 6 and any(c.isdigit() for c in normalized):
        return True
    if (
        len(normalized) >= 8
        and any(c.isalpha() for c in normalized)
        and any(c.isdigit() for c in normalized)
    ):
        return True
    return False


def _find_anchor_value_words(words: List[Dict[str, Any]], label_terms: List[str]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    anchors = _find_label_anchor_boxes(words, label_terms)

    for anchor_box in anchors:
        right_candidates = right_of(anchor_box, words, max_dx=700, same_line_only=True)
        below_candidates = below(anchor_box, words, max_dy=260)

        right_values = [w for w in right_candidates if _looks_like_value_token(str(w.get("text", "")))]
        below_values = [w for w in below_candidates if _looks_like_value_token(str(w.get("text", "")))]

        if right_values:
            matched.extend(right_values[:6])
            continue
        if below_values:
            matched.extend(below_values[:4])
            continue

        # Conservative fallback: take the nearest token to the right if it's likely meaningful text.
        for candidate in right_candidates[:2]:
            txt = str(candidate.get("text", "")).strip()
            if len(normalize_token(txt)) >= 3:
                matched.append(candidate)
                break

    return _dedupe_words_by_box(matched)


def _find_memory_value_words(words: List[Dict[str, Any]], memory_tokens: Dict[str, int]) -> List[Dict[str, Any]]:
    if not memory_tokens:
        return []

    matched: List[Dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        normalized = normalize_token(text)
        if len(normalized) < 5:
            continue

        if normalized in memory_tokens and _should_remember_token(text):
            matched.append(word)
            continue

        # Conservative fuzzy memory match for OCR-noisy long tokens.
        if len(normalized) >= 8:
            for mem_token in memory_tokens.keys():
                if len(mem_token) < 8:
                    continue
                if similarity(normalized, mem_token) >= 0.9 and _should_remember_token(text):
                    matched.append(word)
                    break

    return _dedupe_words_by_box(matched)


def find_pii_words(words: List[Dict[str, Any]], label_synonyms: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    """Find words that contain PII patterns or are likely values near known labels."""
    confident_words = _filter_confident_words(words)
    pii_words: List[Dict[str, Any]] = []

    pii_words.extend(_find_pattern_words(confident_words))

    synonyms_map = label_synonyms or LABEL_SYNONYMS
    for synonyms in synonyms_map.values():
        pii_words.extend(_find_anchor_value_words(confident_words, synonyms))

    memory_tokens = _get_memory_value_tokens()
    if memory_tokens:
        pii_words.extend(_find_memory_value_words(confident_words, memory_tokens))

    return _dedupe_words_by_box(pii_words)

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
        
        # Configure Tesseract from request, env var, or PATH.
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

            runtime_label_synonyms = _load_runtime_label_synonyms()
            
            # Find PII words to mask based on TXT file or fallback to pattern matching
            if txt_found and txt_values:
                # Use TXT-based masking for better accuracy
                pii_words = find_txt_based_pii_words(words, txt_values)
                _remember_successful_matches(txt_values, pii_words)
                print(f"Using TXT-based masking with {len(txt_values)} values, found {len(pii_words)} words to mask")
            else:
                # Fallback to pattern-based masking
                pii_words = find_pii_words(words, runtime_label_synonyms)
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
        
        # Configure Tesseract from request, env var, or PATH.
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
            
            # Use TXT-based masking if values were extracted; otherwise fallback to runtime synonyms.
            if txt_labels:
                pii_words = find_txt_based_pii_words(words, txt_labels)
                _remember_successful_matches(txt_labels, pii_words)
                print(f"Found {len(pii_words)} words to mask based on TXT labels")
            else:
                runtime_label_synonyms = _load_runtime_label_synonyms()
                pii_words = find_pii_words(words, runtime_label_synonyms)
                print(f"TXT labels empty; using fallback pattern/anchor masking with {len(pii_words)} words")
            
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

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model by learning from image+TXT file pairs."""
    try:
        # Check if both files were uploaded
        if 'image_file' not in request.files or 'txt_file' not in request.files:
            return jsonify({'error': 'Both image and TXT files are required for training'}), 400
        
        image_file = request.files['image_file']
        txt_file = request.files['txt_file']
        
        if (image_file.filename == '' or image_file.filename is None or 
            txt_file.filename == '' or txt_file.filename is None):
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid image file type'}), 400
        
        if not txt_file.filename.endswith('.txt'):
            return jsonify({'error': 'TXT file must have .txt extension'}), 400
        
        # Get optional vendor name for better learning
        vendor_name = request.form.get('vendor_name', 'generic').strip() or 'generic'
        tesseract_cmd = request.form.get('tesseract_cmd')
        
        # Configure Tesseract
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
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            words = ocr_words(image)
            
            # Parse TXT file for learning labels/values
            labeled_values = parse_txt_labels(txt_filepath)
            
            if not labeled_values:
                return jsonify({'error': 'No values found in TXT file'}), 400
            
            # Load current memory
            memory = _load_runtime_memory()
            
            # Learn from this pair
            label_synonyms: Dict[str, List[str]] = memory.setdefault("label_synonyms", {})
            vendors: Dict[str, Any] = memory.setdefault("vendors", {})
            value_tokens: Dict[str, Dict[str, int]] = memory.setdefault("value_tokens", {})
            
            vendors.setdefault(vendor_name, {"fields": {}})
            vendor_entry = vendors[vendor_name]
            
            learned_count = 0
            
            # Process each labeled value
            for value in labeled_values:
                tokens = [normalize_token(t) for t in value.split() if normalize_token(t)]
                if not tokens:
                    continue
                
                learned_count += 1
                
                # Record as label synonym if it's short (likely a field name)
                if len(tokens) <= 2:
                    value_str = value.strip()
                    if value_str not in label_synonyms:
                        label_synonyms[value_str] = [value_str]
                    elif value_str not in label_synonyms[value_str]:
                        label_synonyms[value_str].append(value_str)
                    vendor_entry["fields"].setdefault(value_str, 0)
                    vendor_entry["fields"][value_str] += 1
                
                # Learn value token patterns
                value_key = '|'.join(tokens)
                if value_key not in value_tokens:
                    value_tokens[value_key] = {}
                
                for token in tokens:
                    if _should_remember_token(token):
                        if token not in value_tokens[value_key]:
                            value_tokens[value_key][token] = 0
                        value_tokens[value_key][token] += 1

            # Compare annotation values with OCR output and reinforce matched tokens.
            matched_words = find_txt_based_pii_words(words, labeled_values)
            if matched_words:
                _remember_successful_matches(labeled_values, matched_words)
            
            # Save updated memory
            _save_runtime_memory(memory)
            
            # Clean up temporary files
            os.remove(image_filepath)
            os.remove(txt_filepath)
            
            return jsonify({
                'success': True,
                'message': f'Trained on {learned_count} labeled values',
                'vendor': vendor_name,
                'values_learned': learned_count,
                'ocr_words_seen': len(words),
                'ocr_matches': len(matched_words),
                'tokens_captured': len(value_tokens),
            })
            
        except Exception as processing_error:
            # Clean up temporary files on error
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            if os.path.exists(txt_filepath):
                os.remove(txt_filepath)
            raise processing_error
            
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
