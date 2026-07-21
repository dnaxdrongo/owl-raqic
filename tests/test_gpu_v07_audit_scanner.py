import json
import subprocess
import sys


def test_gpu_hotspot_scanner_json_runs():
    out = subprocess.check_output(
        [sys.executable, "scripts/audit_gpu_hotspots.py", "--mode", "json"], text=True
    )
    data = json.loads(out)
    assert isinstance(data, dict)
