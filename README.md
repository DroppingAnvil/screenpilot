# ScreenPilot

A local computer automation agent. Give it a directive, it generates a plan, watches your screen via screenshots, and executes mouse/keyboard actions to complete the task.

## Stack

| Component | Tool |
|-----------|------|
| Planning | Ollama (`llama3.2`) |
| Vision (primary) | Ollama (`qwen2.5vl:7b`) |
| Vision (fallback) | Ollama (`llama3.2-vision`) |
| Screen capture | `mss` |
| Mouse/keyboard | `pyautogui` |
| Memory | SQLite (FTS5) |
| UI | Flask + browser |

---

## Setup

### 1. Python deps

```bash
cd screenpilot
pip install -r requirements.txt
```

On Linux you may also need:
```bash
sudo apt install python3-tk python3-dev scrot
```

### 2. Pull Ollama models

```bash
ollama pull llama3.2
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision   # optional fallback
```

### 3. Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## How it works

```
Directive
   |
   v
Planner (llama3.2) --> Fixed step-by-step plan
   |
   v
Loop:
  Screenshot every 1.5s
     |
     v
  Vision model (qwen2.5vl) reads screen
  Compares visible elements to current plan step
     |
     v
  If step is actionable:
    Vision resolves click coordinates
    Executor clicks/types/scrolls
    Step marked complete
     |
     v
  Memory logs completed step to SQLite
     |
     v
  Advance to next step
```

Memory is searched before each new plan is generated -- completed directives are injected as context so the agent won't repeat past actions (e.g. "already applied for X", "already completed Y survey").

---

## Config (Settings tab in UI)

| Setting | Default | Description |
|---------|---------|-------------|
| Vision primary | `qwen2.5vl:7b` | Best for UI OCR |
| Vision fallback | `llama3.2-vision` | Used if primary fails |
| Planner | `llama3.2` | Text model for step generation |
| Capture interval | `1.5s` | How often to grab the screen |
| Dry Run | off | Logs actions without executing them |
| Type interval | `0.04s` | Delay between keystrokes |

---

## Testing with surveys

1. Open a survey URL in Chrome/Edge
2. In ScreenPilot, type a directive like:
   `Complete the survey at the current tab. Answer honestly. My name is Alex, I am 28, prefer online shopping.`
3. Hit **Generate Plan** -- review the steps
4. Hit **Start** -- watch it go

---

## Project structure

```
screenpilot/
  app.py          Flask server + HTML UI
  agent.py        Main orchestrator loop
  planner.py      Plan generation via Ollama
  vision.py       Screenshot analysis + coord resolution
  executor.py     Mouse/keyboard execution via pyautogui
  capture.py      Screen grab via mss
  memory.py       SQLite memory with FTS search
  requirements.txt
  data/
    screenpilot.db  SQLite database (auto-created)
```

---

## Notes

- pyautogui failsafe: move mouse to top-left corner to immediately abort
- The agent locks the plan on submission -- changing the directive mid-run requires stopping first, then submitting a new directive
- On Windows: no extra deps needed for pyautogui
- On macOS: grant Accessibility and Screen Recording permissions in System Preferences
- ComfyUI integration: not wired in by default since Ollama handles inference directly, but you can point `vision_model_primary` at any Ollama-served model
