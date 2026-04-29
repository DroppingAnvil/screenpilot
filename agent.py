"""
ScreenPilot - Main Agent Orchestrator
"""

import time
import threading
import base64
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from planner import Planner
from vision import VisionModel
from executor import ActionExecutor
from memory import Memory
from capture import ScreenCapture
import ocr as ocr_finder


_VISUAL_ONLY_WORDS = {
    "picture", "photo", "image", "thumbnail", "avatar", "portrait",
    "logo", "icon", "symbol", "graphic", "illustration",
    "video", "gif", "banner", "cover", "artwork",
}


def _is_ocr_searchable(target: str) -> bool:
    """Return False if the target describes a visual/image element that has no OCR-readable text."""
    words = set(target.lower().split())
    return not bool(words & _VISUAL_ONLY_WORDS)


def _clean_ocr_target(raw: str) -> str:
    """Normalize a target_description to plain label text suitable for OCR matching."""
    t = raw.strip()
    # Strip leading action verb + optional article: "Click the X" / "Click X" → "X"
    t = re.sub(r'^(click|tap|select|press|find|open|type in|type)\s+(the\s+|a\s+|an\s+)?', '', t, flags=re.IGNORECASE)
    # Strip remaining leading article
    t = re.sub(r'^(the|a|an)\s+', '', t, flags=re.IGNORECASE)
    # Strip parenthetical qualifiers: "(button)", "(video thumbnail)"
    t = re.sub(r'\s*\([^)]*\)', '', t)
    # Strip trailing positional phrases: "in the top right", "at the bottom of the screen"
    t = re.sub(r'\s+(in|at|on|near)\s+the\s+.+$', '', t, flags=re.IGNORECASE)
    # Strip trailing UI descriptor words that won't appear as OCR text
    t = re.sub(r'\s+(button|icon|link|field|input|box|dropdown|menu|tab|label|checkbox|radio|toggle|bar|logo)s?$', '', t, flags=re.IGNORECASE)
    return t.strip()


class ScreenPilotAgent:
    def __init__(self, config: dict):
        self.config = config
        self.planner = Planner(config)
        self.vision = VisionModel(config)
        self.executor = ActionExecutor(config)
        self.memory = Memory(config["db_path"])
        self.capture = ScreenCapture()

        self.directive = None
        self.negatives = []       # actions/behaviors explicitly blocked by the user
        self.plan = None          # original plan — never modified after generation
        self.working_plan = None  # runtime plan — amended as agent discovers new info
        self.vision_context = "" # task-focused hint for vision, regenerated each step
        self.current_step = 0
        self._last_context_step = -1
        self.running = False
        self.paused = False
        self.status = "idle"
        self.last_action = None
        self.last_screen_b64 = None
        self.last_vision = None
        self.log_entries = []
        self._stuck_count = 0
        self._prev_vision_text = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public control methods (called by Flask routes)                    #
    # ------------------------------------------------------------------ #

    def set_directive(self, directive: str, negatives: list = None) -> dict:
        """Accept a new directive, generate a plan, scan the environment, then refine."""
        self.directive = directive
        self.negatives = negatives or []
        self.status = "planning"
        self._log(f"New directive received: {directive}")
        if self.negatives:
            self._log(f"Blocked actions: {', '.join(self.negatives)}")

        # Check memory for relevant past actions
        past = self.memory.search_relevant(directive)
        past_context = ""
        if past:
            past_context = "\n\nAlready completed (do NOT repeat):\n" + "\n".join(
                f"- {p['summary']}" for p in past
            )

        # Step 1: generate initial plan from directive alone
        plan = self.planner.generate_plan(directive, past_context)
        self._log(f"Initial plan: {len(plan)} steps")

        # Step 2: scan the current screen so the plan reflects reality
        self.status = "scanning"
        self._log("Scanning environment...")
        try:
            screenshot_b64 = self.capture.grab_base64()
            with self._lock:
                self.last_screen_b64 = screenshot_b64
            init_context = self.planner.prepare_vision_context(directive, plan[0] if plan else "")
            vision_result = self.vision.analyze(screenshot_b64, directive, init_context)
            with self._lock:
                self.last_vision = vision_result
            self._log(f"Environment scan complete. Report length: {len(vision_result.get('report', ''))} chars")

            # Step 3: (refine_plan disabled — live reasoning loop adapts to actual screen)
        except Exception as e:
            self._log(f"Environment scan failed, using initial plan: {e}")

        with self._lock:
            self.plan = plan
            self.working_plan = list(plan)
            self.current_step = 0
            self.status = "ready"

        return {"plan": plan, "steps": len(plan)}

    def start(self):
        if not self.plan:
            return {"error": "No plan set. Submit a directive first."}
        self.running = True
        self.paused = False
        self.status = "running"
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        return {"started": True}

    def pause(self):
        self.paused = True
        self.status = "paused"
        return {"paused": True}

    def resume(self):
        self.paused = False
        self.status = "running"
        return {"resumed": True}

    def stop(self):
        self.running = False
        self.status = "stopped"
        return {"stopped": True}

    def get_state(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "directive": self.directive,
                "plan": self.plan or [],
                "working_plan": self.working_plan or [],
                "current_step": self.current_step,
                "last_action": self.last_action,
                "last_vision": self.last_vision,
                "log": self.log_entries[-50:],
                "screen": self.last_screen_b64,
            }

    def set_working_plan(self, steps: list[str]):
        """Replace the working plan (user or runtime edit). Original plan unchanged."""
        with self._lock:
            self.working_plan = list(steps)

    # ------------------------------------------------------------------ #
    #  Core loop                                                          #
    # ------------------------------------------------------------------ #

    def _run_loop(self):
        interval = self.config.get("capture_interval", 1.5)
        max_vision_failures = self.config.get("max_vision_failures", 4)
        consecutive_failures = 0

        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            plan_exhausted = self.current_step >= len(self.working_plan)
            current_step_text = (
                self.working_plan[self.current_step]
                if not plan_exhausted
                else "(plan steps complete — continue based on directive)"
            )
            if plan_exhausted:
                self._log("Plan steps exhausted — running directive-only until task complete")
            else:
                self._log(f"[Step {self.current_step + 1}/{len(self.working_plan)}] Evaluating: {current_step_text}")

            # # Regenerate vision context when step changes (disabled — vision prompt uses no context)
            # if self._last_context_step != self.current_step:
            #     self.vision_context = self.planner.prepare_vision_context(
            #         self.directive, current_step_text
            #     )
            #     self._last_context_step = self.current_step
            #     self._log(f"Vision context: {self.vision_context[:100]}")

            # 1. Capture screen
            screenshot_b64 = self.capture.grab_base64()
            with self._lock:
                self.last_screen_b64 = screenshot_b64

            # 2. Vision: read the screen, describe state and suggest action
            prev_vision = self.last_vision  # capture before overwriting
            vision_result = self.vision.analyze(screenshot_b64, self.directive, self.vision_context)
            with self._lock:
                self.last_vision = vision_result

            report = vision_result.get("report", "")
            self._log(f"Vision: {report[:120]}")

            # Circuit breaker: stop hammering a failing vision model
            if not report:
                consecutive_failures += 1
                if consecutive_failures >= max_vision_failures:
                    self._log(
                        f"Vision model failed {consecutive_failures} times in a row — "
                        "pausing to protect system. Check Ollama and resume when ready."
                    )
                    self.paused = True
                    self.status = "paused"
                    time.sleep(interval)
                    continue
            else:
                consecutive_failures = 0

            if vision_result.get("loading"):
                self._log("Screen still loading, waiting...")
                time.sleep(interval)
                continue

            # Stuck detection: compare current report text to previous
            # If screen text is <50% different, count as stuck iteration
            if self._prev_vision_text:
                similarity = SequenceMatcher(None, self._prev_vision_text, report).ratio()
                if similarity >= 0.5:  # screen hasn't changed meaningfully
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
            self._prev_vision_text = report
            if self._stuck_count > 0:
                self._log(f"Screen unchanged for {self._stuck_count} iteration(s)")

            # 3. Reasoning: feed vision report to text model to decide plan updates + final action
            reason_result = self.planner.reason_about_observation(
                directive=self.directive,
                negatives=self.negatives,
                original_plan=self.plan,
                working_plan=self.working_plan,
                current_step=self.current_step,
                vision_result=vision_result,
                prev_vision_result=prev_vision,
                stuck_count=self._stuck_count,
            )

            reasoning = reason_result.get("reasoning", "")
            self._log(f"Reasoning: {reasoning[:150]}")

            # 4. Apply temp step insertions — deduplicate against existing plan
            insertions = reason_result.get("temp_insertions", [])
            if insertions:
                # Normalize: model sometimes returns dicts instead of plain strings
                flat = []
                for s in insertions:
                    if isinstance(s, str):
                        flat.append(s)
                    elif isinstance(s, dict):
                        # Extract first string value from the dict, ignoring nested objects
                        val = next((v for v in s.values() if isinstance(v, str) and len(v) > 3), None)
                        if val:
                            flat.append(val)
                existing = set(self.working_plan)
                new_steps = [s for s in flat if s.strip() and s not in existing][:2]
                if new_steps:
                    insert_at = self.current_step + 1
                    with self._lock:
                        for i, new_step in enumerate(new_steps):
                            self.working_plan.insert(insert_at + i, new_step)
                    self._log(f"Working plan updated: +{len(new_steps)} temp step(s) inserted")

            # Support both "actions" (list) and legacy "action" (single)
            raw_actions = reason_result.get("actions") or []
            if not raw_actions and reason_result.get("action"):
                raw_actions = [reason_result["action"]]

            result = None
            executed_count = 0

            if not raw_actions or all(a.get("type") == "wait" for a in raw_actions):
                self._log("No action ready, waiting...")
            else:
                for action in raw_actions:
                    if not action or action.get("type") == "wait":
                        continue

                    # 6. Resolve coordinates for click/select actions
                    if action.get("type") in ("click", "select", "double_click", "right_click"):
                        if not action.get("x") or not action.get("y"):
                            raw_target = action.get("target_description", "")
                            if raw_target:
                                target = _clean_ocr_target(raw_target)
                                coords = ocr_finder.find_text_coords(screenshot_b64, target)
                                if not coords:
                                    coords = self.vision.find_coordinates(screenshot_b64, target)
                                action.update(coords)

                    # If a click still has no coords, abort the whole sequence —
                    # subsequent type/tab actions would land in the wrong place
                    if action.get("type") in ("click", "select", "double_click", "right_click"):
                        if not action.get("x") or not action.get("y"):
                            self._log(f"Sequence aborted — no coords for: {action.get('target_description')}")
                            break

                    # 7. Execute
                    result = self.executor.execute(action)
                    self._log(f"Executed: {action.get('type')} -> {result}")
                    self.last_action = action
                    if result and result.get("ok"):
                        executed_count += 1

            # 8. Check if reasoning considers directive complete
            if reason_result.get("directive_complete"):
                self._log("Directive marked complete by reasoning.")
                self.memory.store_completion(self.directive, self.working_plan)
                self.status = "complete"
                self.running = False
                break

            # 9. Advance step if a type action succeeded OR multiple actions ran (form fill sequence)
            #    A lone click on Continue may not have changed the page — don't advance for that alone
            action_types = [a.get("type") for a in raw_actions if a]
            meaningful = executed_count > 0 and (
                len(raw_actions) > 1 or          # multi-action sequence (form fill)
                "type" in action_types or         # text was actually entered
                "key" in action_types or          # keyboard action
                "scroll" in action_types          # scroll happened
            )
            if meaningful and not plan_exhausted:
                self.memory.log_action(
                    directive=self.directive,
                    step=current_step_text,
                    action=self.last_action or {"type": "wait"},
                    result=result or {}
                )
                with self._lock:
                    self.current_step += 1
                self._log(f"{executed_count} action(s) executed. Advancing to step {self.current_step + 1}")

            time.sleep(interval)

    def _log(self, msg: str):
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": msg}
        self.log_entries.append(entry)
        if len(self.log_entries) > 500:
            self.log_entries = self.log_entries[-500:]
        print(f"[{entry['time']}] {msg}")
