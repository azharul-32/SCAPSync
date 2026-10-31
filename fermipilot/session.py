"""Session engine: one SCAPS start does several runs back to back.

Why: starting SCAPS, loading its interface and the model costs about 4 s on a laptop, every time.
Measured on a Ryzen 7 4700U (16 identical runs): one start per run 7.4 s/run; one session for all
16 runs 4.8 s/run; 4 sessions side by side 2.6 s/run. Results are identical to the one-start mode.

Safety: SCAPS runs the script top to bottom and cannot report progress, so a watchdog watches the
result files. Run i is finished when its .iv file exists and has stopped growing (SCAPS writes a
file even for a failed calculation, so a failed run still "finishes"). If run i takes longer than
the time limit, the watchdog kills this SCAPS copy, marks run i as timeout and hands the runs not
yet started back to the queue. Each run's messages from the SCAPS log are kept with that run.
"""
import os
import subprocess
import time

from . import jobs, runner

POLL = 0.25


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def build_session_script(chunk, meas, light):
    lines = ["// FermiPilot session: " + ", ".join(j["run_id"] for j in chunk),
             "set errorhandling.appendtofile", "set script_display_mode.fully_suppressed"]
    for job in chunk:
        body = jobs.build_script(job["run_id"], job["def_path"], job["script_assign"], meas, job["out"], light)
        lines += [b for b in body.split("\r\n")
                  if b and not b.startswith(("set errorhandling", "set script_display", "set quitscript"))]
    lines.append("set quitscript.quitSCAPS")
    return "\r\n".join(lines) + "\r\n"


def run_chunk(scaps_dir, chunk, meas, light, limit, script_path):
    """Run the jobs of `chunk` in one SCAPS session.
    Returns (finished, timed_out_job_or_None, not_started_jobs). finished: list of (job, info)."""
    exe = runner.find_exe(scaps_dir)
    log_path = os.path.join(scaps_dir, "SCAPSErrorLogFile.log")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except OSError:
            pass
    with open(script_path, "w", encoding="latin-1", newline="") as f:
        f.write(build_session_script(chunk, meas, light))

    def log_text():
        try:
            with open(log_path, "rb") as f:
                return f.read().decode("latin-1")
        except OSError:
            return ""

    proc = runner.start_scaps(exe, script_path, scaps_dir)
    finished, i, t_run, log_seen = [], 0, time.time(), 0
    last_size, stable_since = -1, None
    timed_out = None
    while i < len(chunk):
        job = chunk[i]
        iv = job["out"]["iv"]
        size = _size(iv)
        exited = proc.poll() is not None
        if size >= 0 and size == last_size and stable_since and time.time() - stable_since > 0.5 or \
                (exited and size >= 0):
            text = log_text()
            done_at = os.path.getmtime(iv)
            info = {"elapsed_s": round(max(0.0, done_at - t_run), 2), "timed_out": False,
                    "returncode": proc.poll(), "errors": text[log_seen:].strip()}
            log_seen = len(text)
            finished.append((job, info))
            i += 1
            t_run, last_size, stable_since = done_at, -1, None
            continue
        if size != last_size:
            last_size, stable_since = size, time.time()
        if exited:
            break                                  # SCAPS ended before writing this run's result
        if time.time() - t_run > limit:
            timed_out = job
            runner.kill_scaps(proc.pid)
            try:
                proc.kill()
            except OSError:
                pass
            break
        time.sleep(POLL)
    else:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            runner.kill_scaps(proc.pid)
    if timed_out is not None:
        info = {"elapsed_s": round(time.time() - t_run, 2), "timed_out": True, "returncode": None,
                "errors": log_text()[log_seen:].strip()}
        finished.append((timed_out, info))
        rest = chunk[i + 1:]
    else:
        rest = chunk[i:]
    return finished, timed_out, rest
