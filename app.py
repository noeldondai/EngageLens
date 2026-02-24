import os
import sys
import time
import signal
import subprocess
import platform
import re
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "hackathon-demo-secret"

ROOT = Path(__file__).parent.resolve()

# ✅ CHANGE THESE FILENAMES if yours are different
SCRIPTS = {
    "classroom": "webcam_yolo_phone_sleep_tasks_log.py",   # classroom monitor + log
    "home": "webcam_home_study_log.py",                    # home study tracker + log
}

RUNNING = {"name": None, "proc": None}


def _is_windows():
    return platform.system().lower().startswith("win")


def _spawn(script_name: str, args: list[str]) -> subprocess.Popen:
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + args

    if _is_windows():
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            preexec_fn=os.setsid,
        )


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        if _is_windows():
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            time.sleep(0.4)
        else:
            os.killpg(proc.pid, signal.SIGINT)
            time.sleep(0.4)
    except Exception:
        pass

    if proc.poll() is None:
        try:
            proc.terminate()
            time.sleep(0.3)
        except Exception:
            pass

    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


# -------------------------
# LEADERBOARD (reads log file)
# -------------------------
LOG_CANDIDATES = [
    Path("logs/home_study_log.txt"),
    Path("logs/attention_log.txt"),
]

STUDENT_RE = re.compile(r"^Student:\s*(.*?)\s*\|", re.IGNORECASE)
STUDY_RE = re.compile(r"^\s*studying:\s*([0-9]+(\.[0-9]+)?)\s*s", re.IGNORECASE)

def load_leaderboard():
    log_file = None
    for p in LOG_CANDIDATES:
        if p.exists():
            log_file = p
            break
    if log_file is None:
        return None, []

    lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    totals = {}
    current_name = None

    for line in lines:
        m1 = STUDENT_RE.match(line)
        if m1:
            current_name = m1.group(1).strip()
            totals.setdefault(current_name, 0.0)
            continue

        m2 = STUDY_RE.match(line)
        if m2 and current_name:
            totals[current_name] += float(m2.group(1))

    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    rows = [{"rank": i+1, "name": n, "seconds": s} for i, (n, s) in enumerate(ranked)]
    return str(log_file), rows


# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", running=RUNNING["name"])


@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    log_file, rows = load_leaderboard()
    return render_template("leaderboard.html", log_file=log_file, rows=rows)


@app.route("/start/<mode>", methods=["POST"])
def start(mode):
    if mode not in SCRIPTS:
        flash("Unknown mode.")
        return redirect(url_for("index"))

    # stop current if any
    if RUNNING["proc"] is not None and RUNNING["proc"].poll() is None:
        _stop_proc(RUNNING["proc"])
        RUNNING["proc"] = None
        RUNNING["name"] = None

    student_name = request.form.get("student_name", "Noel Karki").strip()
    student_id = request.form.get("student_id", "STU-032").strip()
    period = request.form.get("period", "Day1-Period1").strip()

    # common args (both scripts accept these in our builds)
    args = ["--student_name", student_name, "--student_id", student_id, "--period", period]

    try:
        proc = _spawn(SCRIPTS[mode], args)
        RUNNING["proc"] = proc
        RUNNING["name"] = mode
        flash(f"Started {mode}. (OpenCV window will pop up. Press 'q' there to finish.)")
    except Exception as e:
        flash(f"Failed to start: {e}")

    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop():
    if RUNNING["proc"] is None or RUNNING["proc"].poll() is not None:
        RUNNING["proc"] = None
        RUNNING["name"] = None
        flash("Nothing is running.")
        return redirect(url_for("index"))

    _stop_proc(RUNNING["proc"])
    flash(f"Stopped {RUNNING['name']}.")
    RUNNING["proc"] = None
    RUNNING["name"] = None
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)