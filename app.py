"""
ScreenPilot - Flask Web Server
Serves the control UI and exposes API endpoints for the agent.
"""

import json
from flask import Flask, jsonify, request, render_template_string
from agent import ScreenPilotAgent
from capture import ScreenCapture

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

CONFIG = {
    "ollama_url": "http://localhost:11434",
    "planning_model": "llama3.2",
    "vision_model_primary": "gemma3:4b",
    "vision_timeout": 120,                  # CPU inference ~33s + load headroom
    "capture_interval": 1.5,       # seconds between screen captures
    "db_path": "data/screenpilot.db",
    "dry_run": False,              # set True to test without real mouse clicks
    "type_interval": 0.04,
}

# Auto-detect screen size
_cap = ScreenCapture()
w, h = _cap.get_screen_size()
CONFIG["screen_width"] = w
CONFIG["screen_height"] = h

# ------------------------------------------------------------------ #
#  App + Agent                                                        #
# ------------------------------------------------------------------ #

app = Flask(__name__)
agent = ScreenPilotAgent(CONFIG)

# Pre-load vision model first so it gets full VRAM before llama3.2 loads
import threading
threading.Thread(target=agent.vision.warmup, daemon=True).start()


# ------------------------------------------------------------------ #
#  API Routes                                                         #
# ------------------------------------------------------------------ #

@app.route("/api/directive", methods=["POST"])
def set_directive():
    data = request.get_json() or {}
    directive = data.get("directive", "").strip()
    negatives = data.get("negatives", [])
    if not directive:
        return jsonify({"error": "directive is required"}), 400
    result = agent.set_directive(directive, negatives)
    return jsonify(result)


@app.route("/api/start", methods=["POST"])
def start():
    return jsonify(agent.start())


@app.route("/api/pause", methods=["POST"])
def pause():
    return jsonify(agent.pause())


@app.route("/api/resume", methods=["POST"])
def resume():
    return jsonify(agent.resume())


@app.route("/api/stop", methods=["POST"])
def stop():
    return jsonify(agent.stop())


@app.route("/api/state")
def state():
    return jsonify(agent.get_state())


@app.route("/api/memory/directives")
def memory_directives():
    return jsonify(agent.memory.get_all_directives())


@app.route("/api/memory/log")
def memory_log():
    did = request.args.get("directive_id", type=int)
    return jsonify(agent.memory.get_action_log(directive_id=did))


@app.route("/api/memory/delete/<int:directive_id>", methods=["DELETE"])
def memory_delete(directive_id):
    agent.memory.delete_directive(directive_id)
    return jsonify({"deleted": directive_id})


@app.route("/api/working_plan", methods=["GET", "POST"])
def working_plan():
    if request.method == "POST":
        steps = (request.get_json() or {}).get("steps", [])
        if not isinstance(steps, list):
            return jsonify({"error": "steps must be a list"}), 400
        agent.set_working_plan(steps)
        return jsonify({"updated": True, "steps": len(steps)})
    return jsonify({"working_plan": agent.working_plan or []})


@app.route("/api/config", methods=["GET", "POST"])
def config_route():
    if request.method == "POST":
        updates = request.get_json() or {}
        allowed = {"capture_interval", "dry_run", "planning_model",
                   "vision_model_primary", "type_interval", "vision_timeout"}
        for k, v in updates.items():
            if k in allowed:
                CONFIG[k] = v
                # Propagate to agent submodules
                if k == "dry_run":
                    agent.executor.dry_run = v
                elif k == "capture_interval":
                    agent.config["capture_interval"] = v
        return jsonify({"updated": True, "config": {k: CONFIG[k] for k in allowed}})
    return jsonify({k: CONFIG[k] for k in CONFIG if k != "db_path"})


# ------------------------------------------------------------------ #
#  Frontend                                                           #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    return render_template_string(HTML_UI)


# ------------------------------------------------------------------ #
#  HTML UI                                                            #
# ------------------------------------------------------------------ #

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScreenPilot</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

  :root {
    --bg: #0d0f12;
    --surface: #14171c;
    --surface2: #1c2028;
    --border: #2a2f3a;
    --accent: #00d4aa;
    --accent2: #0099ff;
    --danger: #ff4d6d;
    --warn: #ffb347;
    --text: #e0e6f0;
    --muted: #5a6478;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: grid;
    grid-template-rows: 56px 1fr;
  }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 24px;
    gap: 16px;
  }

  .logo {
    font-family: var(--mono);
    font-weight: 600;
    font-size: 16px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .status-pill {
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s;
  }
  .status-pill.running { color: var(--accent); border-color: var(--accent); box-shadow: 0 0 8px rgba(0,212,170,0.2); }
  .status-pill.planning { color: var(--accent2); border-color: var(--accent2); }
  .status-pill.scanning { color: #c084fc; border-color: #c084fc; }
  .status-pill.paused  { color: var(--warn); border-color: var(--warn); }
  .status-pill.complete { color: var(--accent); border-color: var(--accent); }
  .status-pill.error   { color: var(--danger); border-color: var(--danger); }

  .header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
  .dry-run-badge {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(255,179,71,0.15);
    color: var(--warn);
    border: 1px solid var(--warn);
    display: none;
  }

  main {
    display: grid;
    grid-template-columns: 340px 1fr 300px;
    gap: 0;
    height: calc(100vh - 56px);
    overflow: hidden;
  }

  .panel {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel:last-child { border-right: none; }

  .panel-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
  }
  .panel-body::-webkit-scrollbar { width: 4px; }
  .panel-body::-webkit-scrollbar-track { background: transparent; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* Left panel - controls */
  .directive-input {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    padding: 12px;
    resize: vertical;
    min-height: 90px;
    outline: none;
    transition: border-color 0.2s;
  }
  .directive-input:focus { border-color: var(--accent); }
  .directive-input::placeholder { color: var(--muted); }

  .btn {
    border: none;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    padding: 9px 16px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.15s;
  }
  .btn:active { transform: scale(0.97); }

  .btn-primary { background: var(--accent); color: #000; }
  .btn-primary:hover { background: #00ffe0; }
  .btn-run { background: var(--accent2); color: #fff; }
  .btn-run:hover { background: #3db3ff; }
  .btn-pause { background: rgba(255,179,71,0.15); color: var(--warn); border: 1px solid var(--warn); }
  .btn-stop { background: rgba(255,77,109,0.1); color: var(--danger); border: 1px solid var(--danger); }
  .btn-sm { padding: 5px 10px; font-size: 10px; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }

  .section-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    margin-top: 18px;
  }
  .section-label:first-child { margin-top: 0; }

  /* Plan steps */
  .plan-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 6px;
    margin-bottom: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    font-size: 13px;
    line-height: 1.4;
    transition: all 0.3s;
  }
  .plan-step.active {
    border-color: var(--accent2);
    background: rgba(0,153,255,0.08);
  }
  .plan-step.done {
    border-color: var(--accent);
    background: rgba(0,212,170,0.05);
    color: var(--muted);
  }
  .step-num {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    min-width: 22px;
    padding-top: 1px;
  }
  .plan-step.active .step-num { color: var(--accent2); }
  .plan-step.done .step-num { color: var(--accent); }

  .step-check { font-size: 12px; min-width: 16px; }

  /* Screen preview */
  .screen-preview {
    width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface2);
    aspect-ratio: 16/9;
    object-fit: cover;
    display: block;
  }
  .preview-placeholder {
    width: 100%;
    aspect-ratio: 16/9;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 12px;
  }

  /* Log */
  .log-entry {
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.7;
    color: var(--muted);
    padding: 2px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
  }
  .log-entry .ts { color: var(--border); margin-right: 6px; }
  .log-entry.info { color: var(--text); }
  .log-entry.action { color: var(--accent); }
  .log-entry.warn { color: var(--warn); }

  /* Memory panel */
  .memory-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .memory-item .mem-dir {
    color: var(--text);
    line-height: 1.4;
    margin-bottom: 4px;
  }
  .memory-item .mem-meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
  }
  .memory-item .mem-actions {
    margin-top: 6px;
    display: flex;
    gap: 6px;
  }

  /* Tabs */
  .tabs { display: flex; border-bottom: 1px solid var(--border); }
  .tab {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 16px;
    cursor: pointer;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-content { display: none; flex: 1; overflow: hidden; flex-direction: column; }
  .tab-content.active { display: flex; flex-direction: column; }

  /* Settings */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  .setting-row:last-child { border-bottom: none; }
  .setting-label { color: var(--muted); font-size: 12px; }
  .setting-input {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 4px 8px;
    width: 140px;
    outline: none;
  }
  .setting-input:focus { border-color: var(--accent); }

  .toggle {
    width: 36px; height: 20px;
    background: var(--border);
    border-radius: 20px;
    position: relative;
    cursor: pointer;
    transition: background 0.2s;
  }
  .toggle.on { background: var(--accent); }
  .toggle::after {
    content: '';
    position: absolute;
    width: 14px; height: 14px;
    background: white;
    border-radius: 50%;
    top: 3px; left: 3px;
    transition: left 0.2s;
  }
  .toggle.on::after { left: 19px; }

  .progress-bar-wrap {
    background: var(--surface2);
    border-radius: 4px;
    height: 4px;
    margin-top: 8px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 4px;
    transition: width 0.4s ease;
  }

  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
    display: inline-block;
  }
  .dot.pulse {
    background: var(--accent);
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
  }
</style>
</head>
<body>

<header>
  <span class="logo">ScreenPilot</span>
  <span id="statusPill" class="status-pill">idle</span>
  <span id="dryRunBadge" class="dry-run-badge">DRY RUN</span>
  <div class="header-right">
    <span id="headerDot" class="dot"></span>
    <span style="font-family:var(--mono);font-size:11px;color:var(--muted)" id="stepCounter">—</span>
  </div>
</header>

<main>
  <!-- LEFT: Controls + Plan -->
  <div class="panel">
    <div class="panel-header">
      <span>⬡</span> Control
    </div>
    <div class="panel-body">
      <div class="section-label">Directive</div>
      <textarea id="directiveInput" class="directive-input"
        placeholder="e.g. Complete the customer satisfaction survey at the current URL"></textarea>

      <div class="section-label" style="margin-top:10px">Never Do <span style="font-weight:400;color:var(--muted);font-size:11px">(comma-separated)</span></div>
      <input id="negativesInput" type="text" class="setting-input" style="width:100%;margin-bottom:4px"
        placeholder="e.g. sign in, create account, navigate away, click ads" />

      <div class="btn-row">
        <button class="btn btn-primary" onclick="submitDirective()">Generate Plan</button>
      </div>

      <div class="progress-bar-wrap" id="progressWrap" style="display:none">
        <div class="progress-bar" id="progressBar" style="width:0%"></div>
      </div>

      <div class="btn-row" style="margin-top:14px">
        <button id="btnStart" class="btn btn-run" onclick="sendCmd('start')" disabled>▶ Start</button>
        <button id="btnPause" class="btn btn-pause" onclick="sendCmd('pause')" disabled>⏸ Pause</button>
        <button id="btnResume" class="btn btn-pause" onclick="sendCmd('resume')" style="display:none">↺ Resume</button>
        <button id="btnStop" class="btn btn-stop" onclick="sendCmd('stop')" disabled>■ Stop</button>
      </div>

      <div class="section-label" id="planLabel" style="display:none">Execution Plan</div>
      <div id="planSteps"></div>
    </div>
  </div>

  <!-- CENTER: Screen + Log -->
  <div class="panel">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('screen')">Screen</div>
      <div class="tab" onclick="switchTab('log')">Log</div>
    </div>

    <div id="tabScreen" class="tab-content active">
      <div class="panel-body">
        <div id="screenPreviewWrap">
          <div class="preview-placeholder" id="noScreen">No capture yet</div>
          <img id="screenImg" class="screen-preview" style="display:none" />
        </div>
        <div style="margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--muted)" id="lastActionText">—</div>
      </div>
    </div>

    <div id="tabLog" class="tab-content">
      <div class="panel-body" id="logContainer">
        <div class="log-entry"><span class="ts">—</span>Agent ready.</div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Memory -->
  <div class="panel" style="border-right:none">
    <div class="tabs">
      <div class="tab active" onclick="switchRTab('working')">Working Plan</div>
      <div class="tab" onclick="switchRTab('memory')">Memory</div>
      <div class="tab" onclick="switchRTab('settings')">Settings</div>
    </div>

    <div id="rtabWorking" class="tab-content active">
      <div class="panel-body">
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:8px;line-height:1.5">
          Amends at runtime as the agent discovers new info. Edit freely — original plan is preserved.
        </div>
        <div id="workingPlanSteps"></div>
        <div style="margin-top:10px;display:flex;gap:6px">
          <button class="btn btn-sm btn-primary" onclick="saveWorkingPlan()">Save</button>
          <button class="btn btn-sm" style="background:var(--surface2);color:var(--muted);border:1px solid var(--border)" onclick="resetWorkingPlan()">Reset to Original</button>
        </div>
      </div>
    </div>

    <div id="rtabMemory" class="tab-content">
      <div class="panel-body" id="memoryList">
        <div style="font-family:var(--mono);font-size:11px;color:var(--muted)">No completed directives yet.</div>
      </div>
    </div>

    <div id="rtabSettings" class="tab-content">
      <div class="panel-body">
        <div class="section-label">Models</div>
        <div class="setting-row">
          <span class="setting-label">Vision (primary)</span>
          <input class="setting-input" id="cfgVisionPrimary" value="qwen2.5vl:7b" onchange="saveConfig()">
        </div>
        <div class="setting-row">
          <span class="setting-label">Vision timeout (s)</span>
          <input class="setting-input" id="cfgVisionTimeout" type="number" step="5" min="10" value="90" onchange="saveConfig()">
        </div>
        <div class="setting-row">
          <span class="setting-label">Planner</span>
          <input class="setting-input" id="cfgPlanner" value="llama3.2" onchange="saveConfig()">
        </div>

        <div class="section-label" style="margin-top:18px">Capture</div>
        <div class="setting-row">
          <span class="setting-label">Interval (s)</span>
          <input class="setting-input" id="cfgInterval" type="number" step="0.5" min="0.5" value="1.5" onchange="saveConfig()">
        </div>

        <div class="section-label" style="margin-top:18px">Execution</div>
        <div class="setting-row">
          <span class="setting-label">Dry Run</span>
          <div id="dryRunToggle" class="toggle" onclick="toggleDryRun()"></div>
        </div>
        <div class="setting-row">
          <span class="setting-label">Type interval (s)</span>
          <input class="setting-input" id="cfgTypeInterval" type="number" step="0.01" min="0.01" value="0.04" onchange="saveConfig()">
        </div>
      </div>
    </div>
  </div>
</main>

<script>
let pollInterval = null;
let dryRun = false;

async function submitDirective() {
  const text = document.getElementById('directiveInput').value.trim();
  if (!text) return;
  setStatus('planning');
  document.getElementById('planSteps').innerHTML = '';
  document.getElementById('planLabel').style.display = 'none';
  document.getElementById('progressWrap').style.display = 'none';

  // Poll during planning+scanning so screen capture and status updates appear live
  startPolling();

  const negatives = document.getElementById('negativesInput').value
    .split(',').map(s => s.trim()).filter(Boolean);
  const res = await fetch('/api/directive', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({directive: text, negatives})
  });

  stopPolling();
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  renderPlan(data.plan, 0);
  renderWorkingPlan(data.plan, 0);
  document.getElementById('planLabel').style.display = 'block';
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('btnStart').disabled = false;
  setStatus('ready');
  // One final poll to sync screen capture from the environment scan
  pollState();
}

async function sendCmd(cmd) {
  const res = await fetch(`/api/${cmd}`, {method:'POST'});
  const data = await res.json();
  if (cmd === 'start') {
    startPolling();
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnPause').disabled = false;
    document.getElementById('btnStop').disabled = false;
    document.getElementById('btnPause').style.display = '';
    document.getElementById('btnResume').style.display = 'none';
  }
  if (cmd === 'pause') {
    document.getElementById('btnPause').style.display = 'none';
    document.getElementById('btnResume').style.display = '';
  }
  if (cmd === 'resume') {
    document.getElementById('btnPause').style.display = '';
    document.getElementById('btnResume').style.display = 'none';
  }
  if (cmd === 'stop') {
    stopPolling();
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnPause').disabled = true;
    document.getElementById('btnStop').disabled = true;
  }
}

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(pollState, 1200);
}

function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

async function pollState() {
  const res = await fetch('/api/state');
  const s = await res.json();

  setStatus(s.status);
  updateDot(s.status);

  if (s.plan && s.plan.length) {
    renderPlan(s.plan, s.current_step);
    const pct = s.working_plan && s.working_plan.length
      ? (s.current_step / s.working_plan.length) * 100 : 0;
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('stepCounter').textContent =
      `${s.current_step}/${(s.working_plan || s.plan).length}`;
  }
  if (s.working_plan) renderWorkingPlan(s.working_plan, s.current_step);

  if (s.screen) {
    document.getElementById('noScreen').style.display = 'none';
    const img = document.getElementById('screenImg');
    img.src = 'data:image/png;base64,' + s.screen;
    img.style.display = 'block';
  }

  if (s.last_action) {
    const a = s.last_action;
    let txt = `Last: ${a.type}`;
    if (a.x) txt += ` at (${a.x}, ${a.y})`;
    if (a.text) txt += ` "${a.text.slice(0,30)}"`;
    document.getElementById('lastActionText').textContent = txt;
  }

  if (s.log) renderLog(s.log);

  if (s.status === 'complete' || s.status === 'stopped') {
    stopPolling();
    document.getElementById('btnStop').disabled = true;
    document.getElementById('btnPause').disabled = true;
    document.getElementById('btnStart').disabled = false;
    loadMemory();
  }

  // Circuit-breaker pause: stop hammering the server, show resume button
  if (s.status === 'paused') {
    document.getElementById('btnPause').style.display = 'none';
    document.getElementById('btnResume').style.display = '';
    stopPolling();
  }
}

function renderPlan(steps, currentStep) {
  const el = document.getElementById('planSteps');
  el.innerHTML = steps.map((s, i) => {
    let cls = '';
    let icon = `<span class="step-check"></span>`;
    if (i < currentStep) { cls = 'done'; icon = `<span class="step-check" style="color:var(--accent)">✓</span>`; }
    else if (i === currentStep) { cls = 'active'; icon = `<span class="step-check" style="color:var(--accent2)">▶</span>`; }
    return `<div class="plan-step ${cls}">
      <span class="step-num">${i+1}</span>
      ${icon}
      <span>${s}</span>
    </div>`;
  }).join('');
}

function renderLog(entries) {
  const container = document.getElementById('logContainer');
  container.innerHTML = [...entries].reverse().map(e =>
    `<div class="log-entry"><span class="ts">${e.time}</span>${escHtml(e.msg)}</div>`
  ).join('');
}

function renderWorkingPlan(steps, currentStep) {
  const el = document.getElementById('workingPlanSteps');
  el.innerHTML = steps.map((s, i) => {
    let borderColor = 'var(--border)';
    let textColor = 'var(--text)';
    if (i < currentStep) { borderColor = 'var(--accent)'; textColor = 'var(--muted)'; }
    else if (i === currentStep) { borderColor = 'var(--accent2)'; }
    return `<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:5px">
      <span style="font-family:var(--mono);font-size:10px;color:var(--muted);min-width:18px;padding-top:3px">${i+1}</span>
      <input class="working-step-input" data-index="${i}" value="${escHtml(s)}"
        style="flex:1;background:var(--surface2);border:1px solid ${borderColor};border-radius:4px;
               color:${textColor};font-family:var(--sans);font-size:12px;padding:4px 7px;outline:none"
        onfocus="this.style.borderColor='var(--accent2)'"
        onblur="this.style.borderColor='${borderColor}'" />
      <button onclick="removeWorkingStep(${i})" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;padding:2px 4px;line-height:1">×</button>
    </div>`;
  }).join('') +
  `<button onclick="addWorkingStep()" class="btn btn-sm" style="margin-top:4px;background:var(--surface2);color:var(--muted);border:1px solid var(--border);width:100%">+ Add step</button>`;
}

function getWorkingSteps() {
  return [...document.querySelectorAll('.working-step-input')].map(i => i.value.trim()).filter(Boolean);
}

function addWorkingStep() {
  const steps = getWorkingSteps();
  steps.push('');
  renderWorkingPlan(steps, -1);
  const inputs = document.querySelectorAll('.working-step-input');
  inputs[inputs.length - 1].focus();
}

function removeWorkingStep(idx) {
  const steps = getWorkingSteps();
  steps.splice(idx, 1);
  renderWorkingPlan(steps, -1);
}

async function saveWorkingPlan() {
  const steps = getWorkingSteps();
  await fetch('/api/working_plan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({steps})
  });
}

async function resetWorkingPlan() {
  const state = await (await fetch('/api/state')).json();
  if (state.plan) renderWorkingPlan(state.plan, state.current_step);
  await fetch('/api/working_plan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({steps: state.plan})
  });
}

async function loadMemory() {
  const res = await fetch('/api/memory/directives');
  const items = await res.json();
  const el = document.getElementById('memoryList');
  if (!items.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:11px;color:var(--muted)">No completed directives yet.</div>';
    return;
  }
  el.innerHTML = items.map(m => `
    <div class="memory-item">
      <div class="mem-dir">${escHtml(m.directive)}</div>
      <div class="mem-meta">${m.completed ? '✓ completed' : 'in progress'} · ${(m.completed_at || m.created_at || '').slice(0,10)}</div>
      <div class="mem-actions">
        <button class="btn btn-sm btn-stop" onclick="deleteMemory(${m.id})">Delete</button>
      </div>
    </div>
  `).join('');
}

async function deleteMemory(id) {
  await fetch(`/api/memory/delete/${id}`, {method:'DELETE'});
  loadMemory();
}

function setStatus(s) {
  const el = document.getElementById('statusPill');
  el.textContent = s;
  el.className = 'status-pill ' + s;
}

function updateDot(status) {
  const dot = document.getElementById('headerDot');
  dot.className = 'dot' + (status === 'running' ? ' pulse' : '');
}

function switchTab(name) {
  document.querySelectorAll('.tabs .tab').forEach((t,i) => {
    if (i < 2) t.classList.toggle('active', t.textContent.toLowerCase() === name);
  });
  document.getElementById('tabScreen').classList.toggle('active', name === 'screen');
  document.getElementById('tabLog').classList.toggle('active', name === 'log');
}

function switchRTab(name) {
  const tabs = document.querySelectorAll('.panel:last-child .tab');
  tabs.forEach(t => t.classList.toggle('active', t.textContent.toLowerCase().includes(name.slice(0,4))));
  document.getElementById('rtabWorking').classList.toggle('active', name === 'working');
  document.getElementById('rtabMemory').classList.toggle('active', name === 'memory');
  document.getElementById('rtabSettings').classList.toggle('active', name === 'settings');
}

function toggleDryRun() {
  dryRun = !dryRun;
  document.getElementById('dryRunToggle').classList.toggle('on', dryRun);
  document.getElementById('dryRunBadge').style.display = dryRun ? '' : 'none';
  saveConfig();
}

async function saveConfig() {
  const cfg = {
    vision_model_primary: document.getElementById('cfgVisionPrimary').value,
    planning_model: document.getElementById('cfgPlanner').value,
    capture_interval: parseFloat(document.getElementById('cfgInterval').value) || 1.5,
    type_interval: parseFloat(document.getElementById('cfgTypeInterval').value) || 0.04,
    vision_timeout: parseInt(document.getElementById('cfgVisionTimeout').value) || 90,
    dry_run: dryRun
  };
  await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(cfg)
  });
}

async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  document.getElementById('cfgVisionPrimary').value = cfg.vision_model_primary || 'qwen2.5vl:7b';
  document.getElementById('cfgPlanner').value = cfg.planning_model || 'llama3.2';
  document.getElementById('cfgInterval').value = cfg.capture_interval || 1.5;
  document.getElementById('cfgTypeInterval').value = cfg.type_interval || 0.04;
  document.getElementById('cfgVisionTimeout').value = cfg.vision_timeout || 90;
  dryRun = cfg.dry_run || false;
  document.getElementById('dryRunToggle').classList.toggle('on', dryRun);
  document.getElementById('dryRunBadge').style.display = dryRun ? '' : 'none';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Init
loadMemory();
loadConfig();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    print("ScreenPilot running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
