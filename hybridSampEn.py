"""
ECG vs RR Sample Entropy (hybrid):
- Input CSV (no header): col1 = hand temperature (°C), col2 = ECG (mV)
- ECG SampEn: downsampled (e.g., ds=20), mean-centered, Chebyshev norm, r = 0.2*SD
- RR SampEn: full-rate processing at FS_HZ for detection; SampEn computed on RR intervals (no downsampling)
- Sliding windows at full-rate (WINDOW_SEC, OVERLAP)
- Progress bars like the second script
- Outputs correlations and saves 'windowed_sampen_and_temp_hybrid.csv'
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import time
from pathlib import Path

from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.stats import pearsonr
from scipy.spatial import cKDTree

# Progress bar (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# ======================
# User parameters
# ======================
CSV_PATH   = r"C:\Users\niket\Downloads\simulated_temp_ecg.csv"
FS_HZ      = 2000.0          # ECG sampling rate (Hz)
WINDOW_SEC = 10.0            # window length (s)
OVERLAP    = 0.0             # fraction in [0,1); 0 = non-overlap
M_SAMPEN   = 2
R_FACTOR   = 0.2             # r = 0.2 * SD
DS_ECG     = 20              # <-- downsample factor for ECG SampEn ONLY

# R-peak / detection path settings (stay at full 2 kHz)
ECG_BAND     = (5.0, 35.0)   # bandpass for robust QRS detection
HR_BPM_RANGE = (35, 220)     # plausible HR range
INT_WIN_SEC  = 0.150         # integration window for moving average (s)

# ======================
# Utilities
# ======================
def log(msg):
    print(msg, flush=True)

def bandpass_filter(x, fs, low, high, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def moving_average(x, w_len):
    if w_len <= 1:
        return x.copy()
    kernel = np.ones(w_len, dtype=float) / float(w_len)
    return np.convolve(x, kernel, mode="same")

def robust_thresh(x, k=3.25):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad

def window_indices(N, fs, win_sec, overlap_frac):
    win_len = int(round(win_sec * fs))
    step = int(round(win_len * (1.0 - overlap_frac)))
    if win_len <= 1 or step < 1:
        raise ValueError("Window length or step invalid—adjust WINDOW_SEC/OVERLAP.")
    starts = np.arange(0, N - win_len + 1, step, dtype=int)
    return [(s, s + win_len) for s in starts]

def detect_peaks_pipeline(sig, fs, int_win_sec, hr_range, evaluate_only=False):
    diff = np.ediff1d(sig, to_begin=0)
    sq = diff * diff
    w = max(1, int(round(int_win_sec * fs)))
    integ = moving_average(sq, w)

    thr = robust_thresh(integ, k=3.25)
    hr_min, hr_max = hr_range
    min_dist = int(np.floor(fs * 60.0 / hr_max))  # refractory
    min_dist = max(1, min_dist)

    prominence = 0.25 * (np.percentile(integ, 95) - np.median(integ) + 1e-12)
    peaks, _ = find_peaks(integ, distance=min_dist, height=thr, prominence=prominence)

    if evaluate_only:
        return peaks, integ, thr

    if peaks.size > 1:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] < min_dist:
                if integ[p] > integ[keep[-1]]:
                    keep[-1] = p
            else:
                keep.append(p)
        peaks = np.array(keep, dtype=int)

    return peaks, integ, thr

def refine_to_qrs_max(original_ecg, peaks_integ, fs, search_win_s=0.080):
    half = max(1, int(round(search_win_s * fs)))
    qrs_peaks = []
    for p in peaks_integ:
        i0 = max(0, p - half)
        i1 = min(len(original_ecg), p + half + 1)
        if i1 - i0 < 2:
            continue
        loc = np.argmax(original_ecg[i0:i1])
        qrs_peaks.append(i0 + loc)
    if len(qrs_peaks) == 0:
        return np.array([], dtype=int)
    return np.unique(np.array(qrs_peaks, dtype=int))

def choose_polarity(ecg_f, fs, int_win_sec, hr_range):
    peaks_pos, _, _ = detect_peaks_pipeline(ecg_f, fs, int_win_sec, hr_range, evaluate_only=True)
    peaks_neg, _, _ = detect_peaks_pipeline(-ecg_f, fs, int_win_sec, hr_range, evaluate_only=True)
    return 1.0 if len(peaks_pos) >= len(peaks_neg) else -1.0

# ---- SampEn (Chebyshev, mean-centered). KD-tree impl ----
def _embed(arr, m):
    arr = np.asarray(arr)
    N = arr.size
    if N <= m:
        return np.empty((0, m))
    s0 = arr.strides[0]
    return np.lib.stride_tricks.as_strided(arr, shape=(N - m + 1, m), strides=(s0, s0))

def sampen_kdtree_cheb(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < (m + 2):
        return np.nan
    x = x - np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    rad = r_factor * sd

    Xm  = _embed(x, m)
    Xm1 = _embed(x, m + 1)
    Nm, Nm1 = Xm.shape[0], Xm1.shape[0]
    if Nm < 2 or Nm1 < 2:
        return np.nan

    tree_m  = cKDTree(Xm)
    tree_m1 = cKDTree(Xm1)

    # Chebyshev neighborhoods (p=inf); exclude self by -1
    counts_m  = np.array([len(tree_m.query_ball_point(row, r=rad, p=np.inf))  - 1 for row in Xm])
    counts_m1 = np.array([len(tree_m1.query_ball_point(row, r=rad, p=np.inf)) - 1 for row in Xm1])

    Cm  = counts_m  / max(1, (Nm  - 1))
    Cm1 = counts_m1 / max(1, (Nm1 - 1))
    phi_m  = np.nanmean(Cm)
    phi_m1 = np.nanmean(Cm1)

    if phi_m <= 0 or phi_m1 <= 0 or not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(-np.log(phi_m1 / phi_m))

# ======================
# Main
# ======================
def main():
    t0 = time.time()
    log("Step 1/6: Loading CSV...")
    df = pd.read_csv(CSV_PATH, header=None, names=["hand_temp", "ecg"])
    hand_temp = df["hand_temp"].to_numpy(float)
    ecg_raw   = df["ecg"].to_numpy(float)
    N = len(ecg_raw)
    t = np.arange(N) / FS_HZ
    log(f"  Loaded {N} samples. Duration ≈ {N/FS_HZ:.1f} s")

    # Preprocess for detection (full rate)
    log("Step 2/6: Detrending and bandpass filtering ECG (for detection)...")
    ecg_dt = detrend(ecg_raw)
    ecg_f  = bandpass_filter(ecg_dt, FS_HZ, ECG_BAND[0], ECG_BAND[1])

    # R-peak detection at full rate
    log("Step 3/6: Detecting R-peaks at full rate...")
    pol = choose_polarity(ecg_f, FS_HZ, INT_WIN_SEC, HR_BPM_RANGE)
    ecg_use_det = pol * ecg_f

    peaks_integ, integ, thr = detect_peaks_pipeline(ecg_use_det, FS_HZ, INT_WIN_SEC, HR_BPM_RANGE, evaluate_only=False)
    qrs_peaks = refine_to_qrs_max(ecg_use_det, peaks_integ, FS_HZ, search_win_s=0.080)

    if qrs_peaks.size >= 2:
        rr_all_s = np.diff(qrs_peaks) / FS_HZ
        med_hr = 60.0 / np.median(rr_all_s)
        log(f"  Detected {qrs_peaks.size} R-peaks; median HR ≈ {med_hr:.1f} bpm")
    else:
        log("  WARNING: Too few peaks for robust RR analysis; RR metrics may be NaN.")

    # Sliding windows (full-rate indexing)
    log("Step 4/6: Building windows...")
    win_idxs = window_indices(N, FS_HZ, WINDOW_SEC, OVERLAP)
    num_windows = len(win_idxs)
    log(f"  Number of windows: {num_windows}")

    # Compute SampEns with progress
    log("Step 5/6: Computing SampEn (ECG downsampled, RR full-rate)...")

    sampen_ecg_win = np.empty(num_windows, dtype=float)
    sampen_rr_win  = np.empty(num_windows, dtype=float)
    temp_win_mean  = np.empty(num_windows, dtype=float)
    win_centers_t  = np.empty(num_windows, dtype=float)

    start_time = time.time()
    last_print = start_time

    iterator = tqdm(enumerate(win_idxs), total=num_windows, desc="Windows") if TQDM_AVAILABLE else enumerate(win_idxs)

    for k, (i0, i1) in iterator:
        seg_ecg_full = ecg_raw[i0:i1]         # unfiltered ECG for SampEn (as in the simpler method)
        seg_temp     = hand_temp[i0:i1]
        seg_t        = t[i0:i1]

        # ECG SampEn — downsample only here
        if DS_ECG and DS_ECG > 1:
            seg_ecg_ds = seg_ecg_full[::DS_ECG]
        else:
            seg_ecg_ds = seg_ecg_full
        se_ecg = sampen_kdtree_cheb(seg_ecg_ds, m=M_SAMPEN, r_factor=R_FACTOR)

        # RR SampEn — full-rate detection already done; use peaks within window
        mask_peaks = (qrs_peaks >= i0) & (qrs_peaks < i1)
        seg_peaks = qrs_peaks[mask_peaks]
        if seg_peaks.size >= 3:
            rr_seg_s = np.diff(seg_peaks) / FS_HZ  # RR intervals (s)
            se_rr = sampen_kdtree_cheb(rr_seg_s, m=M_SAMPEN, r_factor=R_FACTOR)
        else:
            se_rr = np.nan

        sampen_ecg_win[k] = se_ecg
        sampen_rr_win[k]  = se_rr
        temp_win_mean[k]  = np.nanmean(seg_temp)
        win_centers_t[k]  = 0.5 * (seg_t[0] + seg_t[-1])

        # Lightweight progress if tqdm not available
        if not TQDM_AVAILABLE:
            now = time.time()
            if now - last_print >= 2.0 or k == num_windows - 1:
                pct = 100.0 * (k + 1) / num_windows
                elapsed = now - start_time
                rate = (k + 1) / elapsed if elapsed > 0 else np.nan
                eta = (num_windows - (k + 1)) / rate if rate and np.isfinite(rate) else np.nan
                log(f"  Window {k+1}/{num_windows} ({pct:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
                last_print = now

    # Correlations vs temperature
    log("Step 6/6: Correlation & output...")
    mask_ecg = np.isfinite(sampen_ecg_win)
    if mask_ecg.sum() >= 3:
        r_ecg, p_ecg = pearsonr(temp_win_mean[mask_ecg], sampen_ecg_win[mask_ecg])
    else:
        r_ecg, p_ecg = (np.nan, np.nan)

    mask_rr = np.isfinite(sampen_rr_win)
    if mask_rr.sum() >= 3:
        r_rr, p_rr = pearsonr(temp_win_mean[mask_rr], sampen_rr_win[mask_rr])
    else:
        r_rr, p_rr = (np.nan, np.nan)

    log("\n=== Correlations (windowed means) ===")
    log(f"Temp vs SampEn(ECG ds{DS_ECG}, Cheb): r = {r_ecg:.3f}, p = {p_ecg:.3g}")
    log(f"Temp vs SampEn(RR, Cheb):            r = {r_rr:.3f}, p = {p_rr:.3g}")

    # Save
    out = pd.DataFrame({
        "time_center_s": win_centers_t,
        "temp_win_mean": temp_win_mean,
        "sampen_ecg_ds": sampen_ecg_win,
        "sampen_rr":     sampen_rr_win
    })
    out.to_csv("windowed_sampen_and_temp_hybrid.csv", index=False)
    log("\nSaved: windowed_sampen_and_temp_hybrid.csv")
    log(f"Done in {time.time() - t0:.1f} s.")

if __name__ == "__main__":
    main()
