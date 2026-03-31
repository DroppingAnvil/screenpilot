"""
ScreenCapture - grabs the primary monitor and returns base64 PNG.
Uses mss for speed (faster than pyautogui for screenshots).
"""

import base64
import io

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image
    import pyautogui
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ScreenCapture:
    def __init__(self, monitor_index: int = 1):
        self.monitor_index = monitor_index

    def grab_base64(self) -> str:
        """Capture screen and return as base64-encoded PNG string."""
        if MSS_AVAILABLE:
            return self._grab_mss()
        elif PIL_AVAILABLE:
            return self._grab_pyautogui()
        else:
            raise RuntimeError("Neither mss nor pyautogui/PIL is installed.")

    def _grab_mss(self) -> str:
        # Create mss context here so Win32 DC handles are initialized on the
        # calling thread. mss uses thread-local storage for srcdc/memdc, so a
        # context created on a different thread will raise AttributeError.
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _grab_pyautogui(self) -> str:
        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_screen_size(self) -> tuple[int, int]:
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                m = sct.monitors[self.monitor_index]
                return m["width"], m["height"]
        elif PIL_AVAILABLE:
            size = pyautogui.size()
            return size.width, size.height
        return 1920, 1080