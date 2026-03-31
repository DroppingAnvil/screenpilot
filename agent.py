"""
ScreenPilot - Main Agent Orchestrator
"""

import time
import threading
import base64
import json
from datetime import datetime
from pathlib import Path

from planner import Planner
from vision import VisionModel
from executor import ActionExecutor
from memory import Memory
from capture import ScreenCapture


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

            # Step 3: refine plan based on what is actually on screen
            plan = self.planner.refine_plan(directive, plan, vision_result)
            self._log(f"Plan refined to {len(plan)} steps")
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

            if self.current_step >= len(self.working_plan):
                self._log("All working plan steps complete.")
                self.memory.store_completion(self.directive, self.working_plan)
                self.status = "complete"
                self.running = False
                break

            current_step_text = self.working_plan[self.current_step]
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

            # 3. Reasoning: feed vision report to text model to decide plan updates + final action
            reason_result = self.planner.reason_about_observation(
                directive=self.directive,
                negatives=self.negatives,
                original_plan=self.plan,
                working_plan=self.working_plan,
                current_step=self.current_step,
                vision_result=vision_result,
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

            action = reason_result.get("action")
            result = None

            if not action or action.get("type") == "wait":
                self._log("No action ready, waiting...")
            else:
                # 6. Resolve coordinates for click actions
                if action.get("type") in ("click", "double_click", "right_click"):
                    if not action.get("x") or not action.get("y"):
                        coords = self.vision.find_coordinates(
                            screenshot_b64,
                            action.get("target_description", current_step_text)
                        )
                        action.update(coords)

                # 7. Execute
                result = self.executor.execute(action)
                self._log(f"Executed: {action.get('type')} -> {result}")
                self.last_action = action

            # 8. Advance step only if an action was executed this iteration
            if result and result.get("ok") and reason_result.get("current_step_complete", False):
                self.memory.log_action(
                    directive=self.directive,
                    step=current_step_text,
                    action=action or {"type": "wait"},
                    result=result or {}
                )
                with self._lock:
                    self.current_step += 1
                self._log(f"Step complete. Advancing to step {self.current_step + 1}")

            time.sleep(interval)

    def _log(self, msg: str):
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": msg}
        self.log_entries.append(entry)
        if len(self.log_entries) > 500:
            self.log_entries = self.log_entries[-500:]
        print(f"[{entry['time']}] {msg}")
