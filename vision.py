"""
Vision - screenshot analysis using multimodal Ollama models.
Primary: qwen2.5vl:7b  |  Fallback: llama3.2-vision
"""

import requests
import json
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

What is on this screen? What text is visible? What can be interacted with? What stands out most? What should happen next?"""


COORD_PROMPT = """Look at this screenshot ({width}x{height} pixels) and find: {target}

First describe where it is:
- Vertical: is it in the top third, middle third, or bottom third of the screen?
- Horizontal: is it on the left, center, or right?
- Is it near any edge?

Then give its center as normalized coordinates (0.0=top/left, 1.0=bottom/right).
Do NOT return (0.5, 0.5) unless the element is literally at the center of the screen.

Reply with ONLY this JSON:
{{"position": "describe where it is", "x": 0.0, "y": 0.0}}"""


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
        prompt = ANALYZE_PROMPT
        raw = self._call_model(prompt, screenshot_b64)
        if not raw:
            return {"report": "", "loading": True}
        print(f"[Vision] Report:\n{raw}")
        return {"report": raw, "loading": False}

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
                nx, ny = result.get("x"), result.get("y")
                if nx is not None and ny is not None:
                    nx, ny = float(nx), float(ny)
                    # Normalized (0.0-1.0) → physical screen pixels
                    # mss and pyautogui both operate in physical pixels on this system
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        x = int(nx * self.screen_width)
                        y = int(ny * self.screen_height)
                    else:
                        x = int(nx)
                        y = int(ny)
                    print(f"[Vision] Normalized ({nx:.3f}, {ny:.3f}) -> screen ({x}, {y})")
                    return {"x": x, "y": y}
        except Exception as e:
            print(f"[Vision] Coord parse error: {e}")

        print(f"[Vision] Could not parse coordinates from: {raw[:200]}")
        return {"x": None, "y": None}

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
