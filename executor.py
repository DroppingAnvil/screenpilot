"""
ActionExecutor - translates action dicts from the vision model into
real mouse and keyboard events via pyautogui.

Supported action types:
  click, double_click, right_click, type, key, scroll, wait, move
"""

import time
import pyautogui

# Safety: pyautogui pause between actions (seconds)
pyautogui.PAUSE = 0.3
# Fail-safe: move mouse to top-left corner to abort
pyautogui.FAILSAFE = True


class ActionExecutor:
    def __init__(self, config: dict):
        self.dry_run = config.get("dry_run", False)
        self.type_interval = config.get("type_interval", 0.04)  # seconds between keystrokes

    def execute(self, action: dict) -> dict:
        if not action:
            return {"ok": False, "reason": "no action"}

        atype = action.get("type", "wait")

        if self.dry_run:
            print(f"[DRY RUN] Would execute: {action}")
            return {"ok": True, "dry_run": True}

        try:
            if atype in ("click", "select"):
                return self._click(action, button="left")
            elif atype == "double_click":
                return self._double_click(action)
            elif atype == "right_click":
                return self._click(action, button="right")
            elif atype == "move":
                return self._move(action)
            elif atype == "type":
                return self._type(action)
            elif atype == "key":
                return self._key(action)
            elif atype == "scroll":
                return self._scroll(action)
            elif atype == "wait":
                duration = action.get("amount", 1)
                time.sleep(duration)
                return {"ok": True, "waited": duration}
            else:
                return {"ok": False, "reason": f"unknown action type: {atype}"}
        except pyautogui.FailSafeException:
            return {"ok": False, "reason": "failsafe triggered (mouse moved to corner)"}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def _click(self, action: dict, button: str = "left") -> dict:
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return {"ok": False, "reason": "missing coordinates"}
        pyautogui.click(int(x), int(y), button=button)
        return {"ok": True, "type": "click", "x": x, "y": y, "button": button}

    def _double_click(self, action: dict) -> dict:
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return {"ok": False, "reason": "missing coordinates"}
        pyautogui.doubleClick(int(x), int(y))
        return {"ok": True, "type": "double_click", "x": x, "y": y}

    def _move(self, action: dict) -> dict:
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return {"ok": False, "reason": "missing coordinates"}
        pyautogui.moveTo(int(x), int(y), duration=0.3)
        return {"ok": True, "type": "move", "x": x, "y": y}

    def _type(self, action: dict) -> dict:
        text = action.get("text", "")
        if not text:
            return {"ok": False, "reason": "no text provided"}
        # Click target first if coordinates provided
        if action.get("x") and action.get("y"):
            pyautogui.click(int(action["x"]), int(action["y"]))
            time.sleep(0.2)
        pyautogui.typewrite(text, interval=self.type_interval)
        return {"ok": True, "type": "type", "text": text[:40]}

    def _key(self, action: dict) -> dict:
        key = action.get("key", "")
        if not key:
            return {"ok": False, "reason": "no key provided"}
        # Support combo keys like "ctrl+a"
        if "+" in key:
            parts = [k.strip() for k in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(key)
        return {"ok": True, "type": "key", "key": key}

    def _scroll(self, action: dict) -> dict:
        x, y = action.get("x"), action.get("y")
        direction = action.get("direction", "down")
        amount = int(action.get("amount", 3))
        clicks = -amount if direction == "down" else amount

        if x and y:
            pyautogui.scroll(clicks, x=int(x), y=int(y))
        else:
            pyautogui.scroll(clicks)
        return {"ok": True, "type": "scroll", "direction": direction, "amount": amount}
