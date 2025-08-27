# run_multi.py
import sys, subprocess
from pathlib import Path

# === Edit these ===
datasets = [
    r"data\Niketh_TEMP_ECG_after200000.csv",
    r"data\Williams-Temp-and-ECG_cols3-2_from54500.csv",
]
fs_hz   = 2000
win_sec = 10
overlap = 0.0
ds_ecg  = 20

# === No edits below ===
here = Path(__file__).parent
script_path = here / "multi_sampen.py"

# Validate paths
missing = [p for p in datasets if not Path(p).exists()]
if not script_path.exists():
    raise FileNotFoundError(f"Cannot find {script_path}")
if missing:
    raise FileNotFoundError(f"Missing dataset(s): {missing}")

cmd = [
    sys.executable,              # <-- use the current python.exe
    str(script_path),
    *datasets,
    "--fs", str(fs_hz),
    "--win-sec", str(win_sec),
    "--overlap", str(overlap),
    "--ds-ecg", str(ds_ecg),
]

print("Running:", " ".join(cmd), flush=True)

# Stream stdout/stderr live; raise on nonzero exit
proc = subprocess.Popen(cmd)
ret = proc.wait()
if ret != 0:
    raise SystemExit(f"multi_sampen.py exited with code {ret}")
