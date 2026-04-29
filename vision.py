"""
Vision - screenshot analysis using multimodal Ollama models.
Primary: qwen2.5vl:7b  |  Fallback: llama3.2-vision
"""

import requests
import json
import re
import base64
import ctypes


def _get_dpi_scale() -> float:
    """Return the Windows DPI scale factor (e.g. 1.25 for 125% scaling)."""
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


ANALYZE_PROMPT = """Describe only what you can see in this screenshot. Do not invent anything.

What is on this screen? What text is visible? What can be interacted with? What stands out most? What should happen next?

Is there a scrollbar visible, or does content appear to be cut off at the bottom of the screen? If so, mention it."""


COORD_PROMPT = """Look at this screenshot ({width}x{height} pixels) and find: {target}

First describe where it is:
- Vertical: is it in the top third, middle third, or bottom third of the screen?
- Horizontal: is it on the left, center, or right?
- Is it near any edge?

Then give its center coordinates in pixels (x: 0=left edge, {width}=right edge | y: 0=top edge, {height}=bottom edge).
Do NOT return the center of the screen unless the element is literally there.

Reply with ONLY this JSON:
{{"position": "describe where it is", "x": 0, "y": 0}}"""


class VisionModel:
    def __init__(self, config: dict):
        self.base_url = config.get("ollama_url", "http://localhost:11434")
        self.model = config.get("vision_model_primary", "qwen2.5vl:7b")
        self.screen_width = config.get("screen_width", 1920)
        self.screen_height = config.get("screen_height", 1080)
        self.dpi_scale = config.get("dpi_scale", _get_dpi_scale())
        self.timeout = config.get("vision_timeout", 120)
        print(f"[Vision] Screen {self.screen_width}x{self.screen_height} @ DPI scale {self.dpi_scale}x")

    def warmup(self):
        """Pre-load the vision model into VRAM without inference.
        Uses /api/chat with no messages so it loads in the right mode."""
        print(f"[Vision] Loading {self.model} into VRAM...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": [], "keep_alive": -1},
                timeout=30
            )
            if resp.status_code == 200:
                print(f"[Vision] {self.model} resident in VRAM.")
            else:
                print(f"[Vision] Warmup returned {resp.status_code}")
        except Exception as e:
            print(f"[Vision] Warmup failed: {e}")

    def analyze(self, screenshot_b64: str, directive: str = "", vision_context: str = "") -> dict:
        raw = self._call_model(ANALYZE_PROMPT, screenshot_b64)
        if not raw:
            return {"report": "", "loading": True, "elements": []}
        print(f"[Vision] Report:\n{raw}")
        return {"report": raw, "loading": False, "elements": []}

    def lookup_element_coords(self, elements: list, target_description: str) -> dict | None:
        """Match target against pre-parsed vision elements. Returns pixel coords or None."""
        if not elements:
            return None

        target_words = set(target_description.lower().split())
        best_match = None
        best_score = 0

        for elem in elements:
            name_words = set(elem["name"].lower().split())
            score = len(target_words & name_words)
            if score > best_score:
                best_score = score
                best_match = elem

        if best_match and best_score > 0:
            print(f"[Vision] Element lookup '{target_description}' -> '{best_match['name']}' at ({best_match['x']}, {best_match['y']})")
            return {"x": best_match["x"], "y": best_match["y"]}

        return None

    def find_coordinates(self, screenshot_b64: str, target_description: str) -> dict:
        prompt = COORD_PROMPT.format(
            target=target_description,
            width=self.screen_width,
            height=self.screen_height,
        )
        raw = self._call_model(prompt, screenshot_b64)
        print(f"[Vision] Coord raw: {raw[:300]}")

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw[start:end])
                x_val, y_val = result.get("x"), result.get("y")
                if x_val is not None and y_val is not None:
                    x, y = float(x_val), float(y_val)
                    # Guard: model may return normalized despite asking for pixels
                    if 0.0 < x <= 1.0 and 0.0 < y <= 1.0:
                        x = int(x * self.screen_width)
                        y = int(y * self.screen_height)
                        print(f"[Vision] Model returned normalized, converted to ({x}, {y})")
                    else:
                        x, y = int(x), int(y)
                        print(f"[Vision] Pixel coords ({x}, {y})")
                    # Sanity check: if coords look like center/edge failure, use position text
                    position_text = result.get("position", "")
                    x, y = self._sanity_check_coords(x, y, position_text)
                    return {"x": x, "y": y}
        except Exception as e:
            print(f"[Vision] Coord parse error: {e}")

        print(f"[Vision] Could not parse coordinates from: {raw[:200]}")
        return {"x": None, "y": None}

    def _sanity_check_coords(self, x: int, y: int, position_text: str) -> tuple:
        """If the model returned center/edge coords (a known failure mode), try to
        estimate from position description text. Returns (None, None) if too ambiguous."""
        w, h = self.screen_width, self.screen_height
        center_x_tol = w * 0.06  # within 6% of center
        center_y_tol = h * 0.06

        is_center = abs(x - w / 2) < center_x_tol and abs(y - h / 2) < center_y_tol
        is_edge = x >= w - 2 or y >= h - 2  # returned resolution as coords

        if not is_center and not is_edge:
            return x, y  # looks fine

        text = position_text.lower()

        # If model explicitly says "center" or "middle", the coords are intentional — trust them
        has_explicit_center = any(t in text for t in ("center", "middle", "centered"))
        has_directional = any(t in text for t in ("top", "bottom", "left", "right", "upper", "lower"))
        if has_explicit_center and not has_directional:
            return x, y

        print(f"[Vision] Suspicious coords ({x},{y}) — estimating from position: '{text}'")

        # Vertical estimate — requires a clear directional word
        if any(t in text for t in ("top", "upper", "header", "toolbar", "address bar", "tab")):
            est_y = int(h * 0.04)   # ~4% from top (browser chrome zone)
        elif "bottom" in text or "lower" in text or "footer" in text:
            est_y = int(h * 0.96)
        else:
            est_y = None  # ambiguous — can't determine vertical position

        # Horizontal estimate — requires a clear directional word
        if "left" in text and "right" not in text:
            est_x = int(w * 0.04)   # ~4% from left
        elif "right" in text and "left" not in text:
            est_x = int(w * 0.96)
        else:
            est_x = None  # ambiguous

        if est_x is None or est_y is None:
            print(f"[Vision] Position too ambiguous to estimate — returning None")
            return None, None

        print(f"[Vision] Position-derived coords: ({est_x}, {est_y})")
        return est_x, est_y

    def _call_model(self, prompt: str, image_b64: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {}
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            body = resp.json()
            content = body.get("message", {}).get("content", "")
            if not content:
                print(f"[Vision] Empty content. Full response: {body}")
            return content
        except Exception as e:
            print(f"[Vision] {self.model} failed: {type(e).__name__}: {e}")
            return ""