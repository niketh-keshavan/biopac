import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, sys

from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.stats import pearsonr
from scipy.spatial import cKDTree

# Try tqdm for a nice progress bar; fall back to simple prints if not installed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# ======================
# User parameters
# ======================
CSV_PATH = "data\Williams-Temp-and-ECG_cols3-2_from54500.csv"  # change if needed
FS_HZ = 2000.0  # <-- critically important: set your ECG sampling rate (Hz)

WINDOW_SEC = 10.0   # sliding window for SampEn
OVERLAP = 0.0       # 0.0 = no overlap
M_SAMPEN = 2
R_FACTOR = 0.2
ECG_BAND = (5.0, 35.0)   # wider band helps robust QRS detection
HR_BPM_RANGE = (35, 220) # plausible HR range for constraints
INT_WIN_SEC = 0.150      # integration window for moving average (s)

# Print helper that always flushes
def log(msg):
    print(msg, flush=True)

# ======================
# Helpers
# ======================
def bandpass_filter(x, fs, low, high, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def moving_average(x, w_len):
    if w_len <= 1:
        return x.copy()
    kernel = np.ones(w_len, dtype=float) / float(w_len)
    return np.convolve(x, kernel, mode="same")

def robust_thresh(x, k=3.5):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad

def embed_ts(x, m):
    x = np.asarray(x)
    N = x.size
    if N <= m:
        return np.empty((0, m))
    s0 = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(N - m + 1, m), strides=(s0, s0))

def sampen_kdtree(x, m=2, r=0.2):
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < (m + 2):
        return np.nan
    std = np.std(x)
    if std == 0:
        return np.nan
    rad = r * std
    Xm = embed_ts(x, m)
    Xm1 = embed_ts(x, m + 1)
    Nm, Nm1 = Xm.shape[0], Xm1.shape[0]
    if Nm < 2 or Nm1 < 2:
        return np.nan
    tree_m = cKDTree(Xm)
    tree_m1 = cKDTree(Xm1)
    counts_m = np.array([len(tree_m.query_ball_point(row, r=rad)) - 1 for row in Xm])
    counts_m1 = np.array([len(tree_m1.query_ball_point(row, r=rad)) - 1 for row in Xm1])
    Cm = counts_m / max(1, (Nm - 1))
    Cm1 = counts_m1 / max(1, (Nm1 - 1))
    phi_m = np.mean(Cm)
    phi_m1 = np.mean(Cm1)
    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan
    return -np.log(phi_m1 / phi_m)

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

    height = thr
    prominence = 0.25 * (np.percentile(integ, 95) - np.median(integ) + 1e-12)
    peaks, props = find_peaks(integ, distance=min_dist, height=height, prominence=prominence)

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

def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

# ======================
# Load data
# ======================
t0 = time.time()
log("Step 1/6: Loading CSV...")
df = pd.read_csv(CSV_PATH, header=None, names=["hand_temp", "ecg"])
hand_temp = df["hand_temp"].to_numpy(float)
ecg_raw = df["ecg"].to_numpy(float)
N = len(ecg_raw)
t = np.arange(N) / FS_HZ
log(f"  Loaded {N} samples. Duration ≈ {N/FS_HZ:.1f} s")

# ======================
# Preprocess ECG
# ======================
log("Step 2/6: Detrending and bandpass filtering ECG...")
ecg = detrend(ecg_raw)
ecg_f = bandpass_filter(ecg, FS_HZ, ECG_BAND[0], ECG_BAND[1])
log("  ECG filtered.")

# ======================
# R-peak detection
# ======================
log("Step 3/6: Detecting R-peaks (adaptive pipeline)...")
pol = choose_polarity(ecg_f, FS_HZ, INT_WIN_SEC, HR_BPM_RANGE)
ecg_use = pol * ecg_f

peaks_integ, integ, thr = detect_peaks_pipeline(ecg_use, FS_HZ, INT_WIN_SEC, HR_BPM_RANGE, evaluate_only=False)
qrs_peaks = refine_to_qrs_max(ecg_use, peaks_integ, FS_HZ, search_win_s=0.080)

if qrs_peaks.size < 5:
    log("  Few peaks; applying fallbacks (abs signal / relaxed threshold)...")
    # Try absolute signal
    peaks_integ2, integ2, thr2 = detect_peaks_pipeline(np.abs(ecg_use), FS_HZ, INT_WIN_SEC, HR_BPM_RANGE, evaluate_only=False)
    qrs_peaks2 = refine_to_qrs_max(ecg_use, peaks_integ2, FS_HZ, search_win_s=0.080)
    if qrs_peaks2.size > qrs_peaks.size:
        qrs_peaks, integ, thr = qrs_peaks2, integ2, thr2

if qrs_peaks.size < 5:
    # Relax threshold (70% of default)
    peaks_integ3, integ3, thr3 = detect_peaks_pipeline(ecg_use, FS_HZ, INT_WIN_SEC, HR_BPM_RANGE, evaluate_only=True)
    lowered = integ3 >= (0.7 * thr3)
    peaks_lo = []
    in_region = False
    start = 0
    for i in range(integ3.size):
        if (integ3[i] >= 0.7 * thr3) and not in_region:
            in_region = True
            start = i
        if in_region and (i == integ3.size-1 or integ3[i] < 0.7 * thr3):
            end = i
            loc = start + np.argmax(integ3[start:end+1])
            peaks_lo.append(loc)
            in_region = False
    qrs_peaks3 = refine_to_qrs_max(ecg_use, np.array(peaks_lo, dtype=int), FS_HZ, search_win_s=0.080)
    if qrs_peaks3.size > qrs_peaks.size:
        qrs_peaks, integ, thr = qrs_peaks3, integ3, thr3

if qrs_peaks.size >= 2:
    rr_all_s = np.diff(qrs_peaks) / FS_HZ
    rr_all_t = t[qrs_peaks[1:]]
    med_hr = 60.0 / np.median(rr_all_s)
    log(f"  Detected {qrs_peaks.size} R-peaks; median HR ≈ {med_hr:.1f} bpm")
else:
    rr_all_s = np.array([])
    rr_all_t = np.array([])
    log("  WARNING: Still too few peaks for RR stats; correlations for RR may be NaN.")

# ======================
# Window setup
# ======================
log("Step 4/6: Building windows...")
win_idxs = window_indices(N, FS_HZ, WINDOW_SEC, OVERLAP)
num_windows = len(win_idxs)
log(f"  Number of windows: {num_windows}")

# ======================
# SampEn (with progress)
# ======================
log("Step 5/6: Computing SampEn for ECG and RR with progress...")

sampen_ecg_win = np.empty(num_windows, dtype=float)
sampen_rr_win  = np.empty(num_windows, dtype=float)
temp_win_mean  = np.empty(num_windows, dtype=float)
win_centers_t  = np.empty(num_windows, dtype=float)

start_time = time.time()
last_print = start_time

if TQDM_AVAILABLE:
    iterator = tqdm(enumerate(win_idxs), total=num_windows, desc="Windows")
else:
    iterator = enumerate(win_idxs)

for k, (i0, i1) in iterator:
    seg_ecg = ecg_use[i0:i1]
    seg_temp = hand_temp[i0:i1]
    seg_t = t[i0:i1]

    # SampEn on ECG waveform segment
    se_ecg = sampen_kdtree(seg_ecg, m=M_SAMPEN, r=R_FACTOR)

    # SampEn on RR within this window
    mask_peaks = (qrs_peaks >= i0) & (qrs_peaks < i1)
    seg_peaks = qrs_peaks[mask_peaks]
    if seg_peaks.size >= 3:
        rr_seg_s = np.diff(seg_peaks) / FS_HZ
        se_rr = sampen_kdtree(rr_seg_s, m=M_SAMPEN, r=R_FACTOR)
    else:
        se_rr = np.nan

    sampen_ecg_win[k] = se_ecg
    sampen_rr_win[k]  = se_rr
    temp_win_mean[k]  = np.nanmean(seg_temp)
    win_centers_t[k]  = 0.5 * (seg_t[0] + seg_t[-1])

    # Lightweight progress if tqdm not available (prints every ~2 seconds)
    if not TQDM_AVAILABLE:
        now = time.time()
        if now - last_print >= 2.0 or k == num_windows - 1:
            pct = 100.0 * (k + 1) / num_windows
            elapsed = now - start_time
            rate = (k + 1) / elapsed if elapsed > 0 else np.nan
            eta = (num_windows - (k + 1)) / rate if rate > 0 else np.nan
            log(f"  Window {k+1}/{num_windows} ({pct:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
            last_print = now

# ======================
# Correlations
# ======================
log("Step 6/6: Correlation & plotting...")
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
if np.isfinite(r_ecg):
    log(f"Temp vs SampEn(ECG waveform): r = {r_ecg:.3f}, p = {p_ecg:.3g}")
else:
    log("Temp vs SampEn(ECG): insufficient data")

if np.isfinite(r_rr):
    log(f"Temp vs SampEn(RR intervals): r = {r_rr:.3f}, p = {p_rr:.3g}")
else:
    log("Temp vs SampEn(RR): insufficient data")

# ======================
# Plots
# ======================
plt.figure(figsize=(12, 4))
plt.plot(t, df["hand_temp"].values, label="Hand Temp")
plt.xlabel("Time (s)")
plt.ylabel("Temperature")
plt.title("Hand Temperature")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.plot(t, ecg_use, label="ECG (filtered, chosen polarity)")
if qrs_peaks.size > 0:
    plt.plot(qrs_peaks / FS_HZ, ecg_use[qrs_peaks], ".", label="R-peaks", markersize=3)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title("ECG with R-peaks")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 4))
plt.plot(win_centers_t, temp_win_mean, label="Temp (win mean)")
plt.plot(win_centers_t, sampen_ecg_win, label="SampEn ECG")
plt.plot(win_centers_t, sampen_rr_win, label="SampEn RR")
plt.xlabel("Time (s)")
plt.title("Windowed Temperature & Sample Entropy")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(5,5))
plt.scatter(temp_win_mean[mask_ecg], sampen_ecg_win[mask_ecg], s=10)
plt.xlabel("Hand Temp (win mean)")
plt.ylabel("SampEn (ECG)")
if np.isfinite(r_ecg):
    plt.title(f"Temp vs SampEn(ECG)\nr={r_ecg:.3f}, p={p_ecg:.3g}")
else:
    plt.title("Temp vs SampEn(ECG)")
plt.tight_layout()

plt.figure(figsize=(5,5))
if mask_rr.sum() >= 3:
    plt.scatter(temp_win_mean[mask_rr], sampen_rr_win[mask_rr], s=10)
else:
    plt.scatter([], [])
plt.xlabel("Hand Temp (win mean)")
plt.ylabel("SampEn (RR)")
if np.isfinite(r_rr):
    plt.title(f"Temp vs SampEn(RR)\nr={r_rr:.3f}, p={p_rr:.3g}")
else:
    plt.title("Temp vs SampEn(RR)")
plt.tight_layout()

plt.show()

# ======================
# Save results
# ======================
out = pd.DataFrame({
    "time_center_s": win_centers_t,
    "temp_win_mean": temp_win_mean,
    "sampen_ecg": sampen_ecg_win,
    "sampen_rr": sampen_rr_win
})
out.to_csv("windowed_sampen_and_temp.csv", index=False)
log("\nSaved: windowed_sampen_and_temp.csv")

log(f"\nDone in {time.time() - t0:.1f} s.")
