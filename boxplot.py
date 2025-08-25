"""
Generate a box plot of ECG "quiet fraction" by temperature quartiles
(Starting from row 59,911; no averaging or downsampling of the ECG amplitude)

What is "quiet fraction"?
- Compute |ΔECG| at the raw sampling rate (absolute difference between consecutive samples).
- Choose a small-change threshold τ as the 30th percentile of all |ΔECG| values in the analyzed segment.
- For each heartbeat (R–R interval), quiet fraction = fraction of raw samples with |ΔECG| < τ.
- Higher quiet fraction ⇒ smoother / less jagged ECG within that beat.

Inputs:
- CSV formatted as exported (quoted header lines), with channels:
  CH1, CH2, CH40, CH41, CH42, CH43
  Where CH2 = ECG (mV), CH40 = Palm Temperature (°C)

Output:
- PNG: quiet_fraction_boxplot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------- Configuration --------------------------- #
CSV_PATH = Path("data/Williams-Temp-and-ECG.csv")  # <-- update if needed
START_ROW_1BASED = 59911                      # start from this 1-based row index
FS = 2000                                     # Hz (0.5 ms/sample)
QUIET_PERCENTILE = 30                         # τ is the 30th percentile of |ΔECG|
SMOOTH_MS = 30                                # ms for smoothing derivative energy
REFRACTORY_S = 0.25                           # s refractory in peak detection
PERCENTILES_FOR_THRESH = [98, 96, 94, 92, 90] # adaptive thresholds for peaks
OUTPUT_PNG = "quiet_fraction_boxplot.png"

# --------------------------- Helpers --------------------------- #
def parse_custom_csv(csv_path: Path, start_row_1based: int):
    """
    Parse the custom exported CSV:
    - Find the channel-header line: "CH1,CH2,CH40,CH41,CH42,CH43,"
    - Skip the "samples" line below it
    - From the requested row onward, extract CH2 (ECG mV) and CH40 (Temp °C)
    Returns: ecg (np.ndarray), temp (np.ndarray)
    """
    with open(csv_path, "r", errors="replace") as f:
        lines = f.readlines()

    # Locate the data header
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('"CH1,CH2,CH40,CH41,CH42,CH43'):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Channel header line not found in CSV.")

    data_start = header_idx + 2  # next line is sample counts; actual data starts after that
    start_row_0based = int(start_row_1based) - 1

    ecg_vals, temp_vals = [], []
    row_idx = 0
    for line in lines[data_start:]:
        s = line.strip()
        if not s.startswith('"') or "," not in s:
            continue
        parts = s.strip('"').split(",")
        # Drop trailing empty field if present (due to trailing comma)
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) < 6:
            continue
        if row_idx >= start_row_0based:
            try:
                ecg_mv = float(parts[1])  # CH2
                temp_c = float(parts[2])  # CH40
                ecg_vals.append(ecg_mv)
                temp_vals.append(temp_c)
            except ValueError:
                pass
        row_idx += 1

    if len(ecg_vals) == 0:
        raise RuntimeError("No numeric data parsed from the requested start row.")

    return np.asarray(ecg_vals, float), np.asarray(temp_vals, float)

def detect_r_peaks(ecg: np.ndarray, fs: int, smooth_ms: float = 30.0,
                   refractory_s: float = 0.25, percentiles=(98,96,94,92,90)):
    """
    Simple R-like peak detection using smoothed derivative-squared energy.
    Returns: r_idx (np.ndarray of sample indices for detected peaks)
    """
    d = np.diff(ecg)
    e = d * d
    w = max(3, int((smooth_ms / 1000.0) * fs))
    kern = np.ones(w, dtype=float) / w
    es = np.convolve(e, kern, mode="same")

    min_dist = int(refractory_s * fs)

    def pick_peaks(eng, thr):
        N = eng.size
        peaks = []
        last = -10**9
        for i in range(1, N - 1):
            if eng[i] > thr and eng[i] > eng[i - 1] and eng[i] >= eng[i + 1] and (i - last) >= min_dist:
                peaks.append(i)
                last = i
        return np.array(peaks, dtype=int)

    # Try descending thresholds until we find enough peaks
    chosen_thr = None
    peaks = np.array([], dtype=int)
    for p in percentiles:
        thr = np.percentile(es, p)
        peaks = pick_peaks(es, thr)
        if peaks.size >= 100:  # heuristic: enough beats to be plausible
            chosen_thr = thr
            break
    if peaks.size == 0:
        # fallback: pick the global maximum
        peaks = np.array([int(np.argmax(es))], dtype=int)
        chosen_thr = float(np.max(es))

    # Map derivative-energy index to ECG sample index (+1)
    r_idx = peaks + 1

    # Clip to valid range (avoid edges)
    r_idx = r_idx[(r_idx > 0) & (r_idx < (ecg.size - 1))]
    r_idx.sort()
    return r_idx, chosen_thr

def compute_quiet_fraction_per_beat(ecg: np.ndarray, temp: np.ndarray, fs: int,
                                    tau_percentile: float = 30.0):
    """
    Compute beat-wise quiet fraction and per-beat median temperature.
    - tau is chosen as the global percentile of |ΔECG| across the segment.
    Returns:
      beat_df: DataFrame with beat-wise metrics (RR_s, Temp_C_med, QuietFraction, etc.)
      tau: float threshold used
    """
    d1 = np.abs(np.diff(ecg))
    tau = float(np.nanpercentile(d1, tau_percentile))

    r_idx, _ = detect_r_peaks(ecg, fs=fs)
    # Discard the very first/last peaks for safe intervals
    if r_idx.size < 3:
        raise RuntimeError("Too few R-peaks detected to form beat intervals.")
    r_idx = r_idx[(r_idx > 0) & (r_idx < (ecg.size - 1))]

    rows = []
    for i in range(r_idx.size - 1):
        a, b = int(r_idx[i]), int(r_idx[i + 1])
        if b <= a + 10:
            continue
        # d1[j] corresponds to change from sample j to j+1 ⇒ use d1[a:b]
        seg_d1 = d1[a:b]
        seg_temp = temp[a:b]

        if seg_d1.size == 0 or seg_temp.size == 0:
            continue

        quiet_fraction = float(np.mean(seg_d1 < tau))
        temp_med = float(np.median(seg_temp))
        rr_s = (b - a) / fs

        rows.append((i, a, b, rr_s, temp_med, quiet_fraction))

    beat_df = pd.DataFrame(rows, columns=[
        "beat_idx", "start_idx", "end_idx", "RR_s", "Temp_C_med", "QuietFraction"
    ])
    return beat_df, tau

def make_boxplot_by_temp_quartiles(beat_df: pd.DataFrame, outfile: str):
    """
    Create a box plot of QuietFraction grouped by temperature quartiles (per-beat median temp).
    Saves the figure to `outfile`.
    """
    # Bin by quartiles of per-beat median temperature
    qlabels = ["Q1 (coolest)", "Q2", "Q3", "Q4 (warmest)"]
    quart = pd.qcut(beat_df["Temp_C_med"], 4, labels=qlabels)
    data_by_q = [beat_df.loc[quart == lab, "QuietFraction"].values for lab in qlabels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data_by_q, labels=qlabels, showfliers=False)
    plt.ylabel("Quiet fraction per beat  (|ΔECG| < τ)")
    plt.title("ECG 'quiet fraction' by temperature quartiles\n(per-beat medians; no averaging of ECG amplitude)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()  # uncomment if you want to display interactively

# --------------------------- Main --------------------------- #
if __name__ == "__main__":
    # 1) Load ECG (mV) and Temp (°C) starting at the requested row
    ecg, temp = parse_custom_csv(CSV_PATH, START_ROW_1BASED)

    # 2) Compute beat-wise quiet fraction + per-beat median temperature
    beat_df, tau = compute_quiet_fraction_per_beat(ecg, temp, fs=FS, tau_percentile=QUIET_PERCENTILE)

    # 3) Make the box plot grouped by temperature quartiles
    make_boxplot_by_temp_quartiles(beat_df, OUTPUT_PNG)

    # 4) Print a small summary so you know what was used
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Beats used: {len(beat_df)}")
    print(f"τ (quiet threshold): {tau:.6f} mV/sample")
    print("Quartile counts:")
    qlabels = ["Q1 (coolest)", "Q2", "Q3", "Q4 (warmest)"]
    quart = pd.qcut(beat_df["Temp_C_med"], 4, labels=qlabels)
    print(pd.value_counts(quart, sort=False).to_string())
