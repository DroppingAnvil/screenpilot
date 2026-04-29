"""
OCR-based element finder — locates UI text in screenshots without a VLM call.
Uses easyocr (pip install easyocr). Falls back silently if not installed.
"""
import base64
import io

_reader = None
OCR_AVAILABLE = False

try:
    import easyocr
    import numpy as np
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    pass


def _get_reader():
    global _reader
    if _reader is None:
        print("[OCR] Initializing easyocr (first run downloads ~50MB of models)...")
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OCR] Reader ready.")
    return _reader


def find_text_coords(screenshot_b64: str, target: str | None, threshold: float = 0.65) -> dict | None:
    """
    Find target text in screenshot.
    Returns {"x": int, "y": int} in physical pixels, or None if not found.

    Tries exact substring match first, then word-overlap scored by OCR confidence.
    """
    if not OCR_AVAILABLE or not target:
        return None

    try:
        img_bytes = base64.b64decode(screenshot_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        results = _get_reader().readtext(img_array)
        # Each result: ([[tl, tr, br, bl], text, confidence)

        target_lower = target.lower()
        target_words = set(target_lower.split())
        best_match = None
        best_score = 0.0

        for (bbox, text, conf) in results:
            text_lower = text.lower()

            if text_lower == target_lower:
                # Exact full match — best possible
                score = conf
            elif len(text_lower) > 3 and (target_lower in text_lower or text_lower in target_lower):
                # Substring match — penalize by length ratio so "Continue" loses to "Accept and Continue"
                length_ratio = min(len(text_lower), len(target_lower)) / max(len(text_lower), len(target_lower))
                score = conf * length_ratio
            else:
                text_words = set(text_lower.split())
                overlap = len(target_words & text_words)
                if overlap == 0:
                    continue
                score = (overlap / max(len(target_words), 1)) * conf

            if score > best_score:
                best_score = score
                best_match = (bbox, text, conf)

        if best_match and best_score >= threshold:
            bbox = best_match[0]
            x = int((bbox[0][0] + bbox[2][0]) / 2)
            y = int((bbox[0][1] + bbox[2][1]) / 2)
            print(f"[OCR] '{target}' -> '{best_match[1]}' (score={best_score:.2f}) at ({x}, {y})")
            return {"x": x, "y": y}

        print(f"[OCR] No match for '{target}' (best score={best_score:.2f})")

    except Exception as e:
        print(f"[OCR] Error: {type(e).__name__}: {e}")

    return None