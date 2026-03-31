"""
Vision - screenshot analysis using multimodal Ollama models.
Primary: qwen2.5vl:7b  |  Fallback: llama3.2-vision
"""

import requests
import json
import base64


ANALYZE_PROMPT = """GOAL: {vision_context}

Describe only what you can see in this screenshot. Do not invent anything.

What is on this screen? What text is visible? What can be interacted with? What stands out most? What should happen next?"""


COORD_PROMPT = """Find the center pixel coordinates of this UI element in the screenshot: {target}

Reply with only this JSON: {{"x": <pixels from left edge>, "y": <pixels from top edge>}}
Image dimensions: {width}x{height} pixels."""


class VisionModel:
    def __init__(self, config: dict):
        self.base_url = config.get("ollama_url", "http://localhost:11434")
        self.model = config.get("vision_model_primary", "qwen2.5vl:7b")
        self.screen_width = config.get("screen_width", 1920)
        self.screen_height = config.get("screen_height", 1080)
        self.timeout = config.get("vision_timeout", 120)

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
        prompt = ANALYZE_PROMPT.format(vision_context=vision_context or "Complete the task visible on screen.")
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

        # Parse JSON pixel coords {"x": 123, "y": 456}
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw[start:end])
                x, y = result.get("x"), result.get("y")
                if x is not None and y is not None:
                    print(f"[Vision] Coords -> ({x}, {y})")
                    return {"x": int(x), "y": int(y)}
        except Exception:
            pass

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
