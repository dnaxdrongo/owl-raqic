import os
import subprocess
import sys


def test_preflight_script_imports_without_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    cp = subprocess.run(
        [sys.executable, "scripts/gpu_v08_preflight.py", "--out", str(tmp_path)],
        cwd=".",
        env=env,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr
    assert (tmp_path / "gpu_v08_preflight.json").exists()
