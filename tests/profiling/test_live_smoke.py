from __future__ import annotations

"""Optional live profiling smoke test that interrupts a real run after a short window.

Enable by setting PROFILE_LIVE_SMOKE_CMD to a full profiling command. The test lets the
process run briefly (default 5s) to surface early failures, then delivers SIGINT to
mirror a keyboard interrupt.
"""

import os
import shlex
import signal
import subprocess
import time

import pytest

SMOKE_CMD_ENV = "PROFILE_LIVE_SMOKE_CMD"
SMOKE_TIMEOUT_ENV = "PROFILE_LIVE_SMOKE_TIMEOUT"
SMOKE_GRACE_ENV = "PROFILE_LIVE_SMOKE_GRACE"
SMOKE_CWD_ENV = "PROFILE_LIVE_SMOKE_CWD"

pytestmark = pytest.mark.skipif(
    not os.getenv(SMOKE_CMD_ENV),
    reason=f"set {SMOKE_CMD_ENV} to run the live profiling smoke test",
)


def _read_output(proc: subprocess.Popen) -> str:
    if proc.stdout is None:
        return ""
    try:
        return proc.stdout.read()
    except Exception:
        return ""


def test_profiling_live_smoke_interrupts_after_grace():
    cmd_raw = os.environ[SMOKE_CMD_ENV]
    cmd = shlex.split(cmd_raw)
    cwd = os.getenv(SMOKE_CWD_ENV) or None
    grace_sec = int(os.getenv(SMOKE_TIMEOUT_ENV, "5"))
    post_interrupt_grace = int(os.getenv(SMOKE_GRACE_ENV, "10"))

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    start = time.time()
    time.sleep(grace_sec)

    if proc.poll() is not None:
        output = _read_output(proc)
        pytest.fail(
            f"Process exited before the interrupt window (code {proc.returncode}). "
            f"Output:\n{output}"
        )

    if hasattr(os, "killpg") and proc.pid:
        os.killpg(proc.pid, signal.SIGINT)
    else:
        proc.send_signal(signal.SIGINT)

    try:
        output, _ = proc.communicate(timeout=post_interrupt_grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        pytest.fail(f"Process did not exit after SIGINT. Partial output:\n{output}")

    runtime = time.time() - start
    assert runtime >= grace_sec, f"process terminated too early ({runtime:.2f}s)"
    assert proc.returncode in (-signal.SIGINT, 130), (
        f"unexpected return code: {proc.returncode}"
    )
