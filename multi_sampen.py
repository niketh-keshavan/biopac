"""
Multi-dataset ECG/RR Sample Entropy analysis with meta-analysis.

Input CSVs (no header): col1 = hand temperature (°C), col2 = ECG (mV)

Per dataset:
  - ECG SampEn: downsampled by DS_ECG, mean-centered, Chebyshev norm, r = 0.2*SD
  - RR SampEn: full-rate detection at FS_HZ; SampEn on RR intervals (no downsampling)
  - Sliding windows of length WINDOW_SEC, overlap OVERLAP
  - Pearson & Spearman correlations vs window-mean temperature
  - Saves "<stem>__windowed_sampen_and_temp.csv"

Across datasets:
  - Fixed-effect meta-analysis (Fisher z) for Pearson r (ECG & RR separately)
  - Saves "summary_meta_analysis.csv"

Usage:
  python multi_sampen.py <csv1> <csv2> ... [--fs 2000] [--win-sec 10] [--overlap 0.0] [--ds-ecg 20]

Notes:
  - Only ECG SampEn is downsampled. RR pipeline runs at full sampling rate.
  - Fisher z meta-analysis assumes independent datasets of the same design.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import glob
import time
import math
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.stats import pearsonr, spearmanr, norm
from scipy.spatial import cKDTree

# Progress bar (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# -----------------------
# Utilities
# -----------------------
def log(msg: str) -> None:
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

# ---- SampEn (Chebyshev, mean-centered) ----
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

    counts_m  = np.array([len(tree_m.query_ball_point(row, r=rad, p=np.inf))  - 1 for row in Xm])
    counts_m1 = np.array([len(tree_m1.query_ball_point(row, r=rad, p=np.inf)) - 1 for row in Xm1])

    Cm  = counts_m  / max(1, (Nm  - 1))
    Cm1 = counts_m1 / max(1, (Nm1 - 1))
    phi_m  = np.nanmean(Cm)
    phi_m1 = np.nanmean(Cm1)

    if phi_m <= 0 or phi_m1 <= 0 or not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(-np.log(phi_m1 / phi_m))

def corr_pearson_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 3:
        return dict(n=n, r=np.nan, p=np.nan, rho=np.nan, p_rho=np.nan)
    r, p = pearsonr(x, y)
    try:
        rho, p_rho = spearmanr(x, y)
    except Exception:
        rho, p_rho = (np.nan, np.nan)
    return dict(n=n, r=float(r), p=float(p), rho=float(rho), p_rho=float(p_rho))

# ---- Fisher z meta-analysis for Pearson r (fixed-effect) ----
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))

def inv_fisher_z(z):
    return np.tanh(z)

def meta_fixed_effect_pearson(rs, ns):
    """
    rs: array-like of Pearson r per dataset
    ns: array-like of corresponding sample sizes (number of windows used)
    Returns: dict with r_fixed, z, se_z, z_stat, p, ci_low, ci_high
    """
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)
    mask = np.isfinite(rs) & np.isfinite(ns) & (ns > 3)
    rs, ns = rs[mask], ns[mask]
    if rs.size == 0:
        return dict(r_fixed=np.nan, z=np.nan, se_z=np.nan, z_stat=np.nan, p=np.nan,
                    ci_low=np.nan, ci_high=np.nan, k=0, n_total=0)
    zs = fisher_z(rs)
    ws = ns - 3.0  # classic weighting
    z_fixed = np.sum(ws * zs) / np.sum(ws)
    se = 1.0 / np.sqrt(np.sum(ws))
    z_stat = z_fixed / se
    p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    # 95% CI on z-scale, then back-transform
    z_lo = z_fixed - 1.96 * se
    z_hi = z_fixed + 1.96 * se
    r_fixed = inv_fisher_z(z_fixed)
    ci_low = inv_fisher_z(z_lo)
    ci_high = inv_fisher_z(z_hi)
    return dict(r_fixed=float(r_fixed), z=float(z_fixed), se_z=float(se),
                z_stat=float(z_stat), p=float(p),
                ci_low=float(ci_low), ci_high=float(ci_high),
                k=int(rs.size), n_total=int(np.sum(ns)))

# -----------------------
# Core per-dataset run
# -----------------------
def analyze_dataset(csv_path: Path,
                    fs_hz: float,
                    window_sec: float,
                    overlap: float,
                    ds_ecg: int,
                    m_sampen: int = 2,
                    r_factor: float = 0.2,
                    ecg_band=(5.0, 35.0),
                    hr_bpm_range=(35, 220),
                    int_win_sec=0.150):
    t0 = time.time()
    stem = csv_path.stem
    log(f"\n=== Dataset: {csv_path} ===")
    df = pd.read_csv(csv_path, header=None, names=["hand_temp", "ecg"])
    hand_temp = df["hand_temp"].to_numpy(float)
    ecg_raw   = df["ecg"].to_numpy(float)
    N = len(ecg_raw)
    t = np.arange(N) / fs_hz
    log(f"  Samples: {N}  |  Duration: {N/fs_hz:.1f}s")

    # Full-rate preprocessing for detection
    ecg_dt = detrend(ecg_raw)
    ecg_f  = bandpass_filter(ecg_dt, fs_hz, ecg_band[0], ecg_band[1])

    # R-peak detection (full-rate)
    pol = choose_polarity(ecg_f, fs_hz, int_win_sec, hr_bpm_range)
    ecg_use_det = pol * ecg_f
    peaks_integ, _, _ = detect_peaks_pipeline(ecg_use_det, fs_hz, int_win_sec, hr_bpm_range, evaluate_only=False)
    qrs_peaks = refine_to_qrs_max(ecg_use_det, peaks_integ, fs_hz, search_win_s=0.080)

    if qrs_peaks.size >= 2:
        rr_all_s = np.diff(qrs_peaks) / fs_hz
        med_hr = 60.0 / np.median(rr_all_s)
        log(f"  R-peaks: {qrs_peaks.size}  |  median HR ≈ {med_hr:.1f} bpm")
    else:
        log("  WARNING: Few peaks; RR metrics may be NaN.")

    # Windows
    win_idxs = window_indices(N, fs_hz, window_sec, overlap)
    num_windows = len(win_idxs)
    log(f"  Windows: {num_windows} (len {window_sec}s, overlap {overlap*100:.0f}%)")

    # Allocate
    sampen_ecg_win = np.empty(num_windows, dtype=float)
    sampen_rr_win  = np.empty(num_windows, dtype=float)
    temp_win_mean  = np.empty(num_windows, dtype=float)
    win_centers_t  = np.empty(num_windows, dtype=float)

    iterator = tqdm(enumerate(win_idxs), total=num_windows, desc=f"{stem}") if TQDM_AVAILABLE else enumerate(win_idxs)
    start_time = time.time()
    last_print = start_time

    for k, (i0, i1) in iterator:
        seg_ecg_full = ecg_raw[i0:i1]     # ECG for SampEn (unfiltered)
        seg_temp     = hand_temp[i0:i1]
        seg_t        = t[i0:i1]

        # ECG SampEn — downsample ONLY here
        seg_ecg_ds = seg_ecg_full[::ds_ecg] if (ds_ecg and ds_ecg > 1) else seg_ecg_full
        se_ecg = sampen_kdtree_cheb(seg_ecg_ds, m=m_sampen, r_factor=r_factor)

        # RR SampEn — from peaks within window (full-rate detection already done)
        mask_peaks = (qrs_peaks >= i0) & (qrs_peaks < i1)
        seg_peaks = qrs_peaks[mask_peaks]
        if seg_peaks.size >= 3:
            rr_seg_s = np.diff(seg_peaks) / fs_hz
            se_rr = sampen_kdtree_cheb(rr_seg_s, m=m_sampen, r_factor=r_factor)
        else:
            se_rr = np.nan

        sampen_ecg_win[k] = se_ecg
        sampen_rr_win[k]  = se_rr
        temp_win_mean[k]  = np.nanmean(seg_temp)
        win_centers_t[k]  = 0.5 * (seg_t[0] + seg_t[-1])

        if not TQDM_AVAILABLE:
            now = time.time()
            if now - last_print >= 2.0 or k == num_windows - 1:
                pct = 100.0 * (k + 1) / num_windows
                elapsed = now - start_time
                rate = (k + 1) / max(elapsed, 1e-9)
                eta = (num_windows - (k + 1)) / rate if rate > 0 else np.nan
                log(f"    {stem}: {k+1}/{num_windows} ({pct:.1f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
                last_print = now

    # Correlations vs temperature
    stats_ecg = corr_pearson_spearman(temp_win_mean, sampen_ecg_win)
    stats_rr  = corr_pearson_spearman(temp_win_mean, sampen_rr_win)

    log("  Correlations (Pearson):")
    log(f"    Temp vs SampEn(ECG ds, Cheb): r = {stats_ecg['r']:.3f}, p = {stats_ecg['p']:.3g} (n={stats_ecg['n']})")
    log(f"    Temp vs SampEn(RR,  Cheb):     r = {stats_rr['r']:.3f}, p = {stats_rr['p']:.3g} (n={stats_rr['n']})")

    # Save per-dataset windows
    out = pd.DataFrame({
        "time_center_s": win_centers_t,
        "temp_win_mean": temp_win_mean,
        "sampen_ecg_ds": sampen_ecg_win,
        "sampen_rr":     sampen_rr_win
    })
    out_name = f"{stem}__windowed_sampen_and_temp.csv"
    out.to_csv(out_name, index=False)
    log(f"  Saved: {out_name}  |  Done in {time.time() - t0:.1f}s")

    return dict(
        path=str(csv_path),
        n_ecg=int(stats_ecg["n"]),
        r_ecg=float(stats_ecg["r"]),
        p_ecg=float(stats_ecg["p"]),
        rho_ecg=float(stats_ecg["rho"]),
        p_rho_ecg=float(stats_ecg["p_rho"]),
        n_rr=int(stats_rr["n"]),
        r_rr=float(stats_rr["r"]),
        p_rr=float(stats_rr["p"]),
        rho_rr=float(stats_rr["rho"]),
        p_rho_rr=float(stats_rr["p_rho"]),
    )

# --- Random-effects (DerSimonian–Laird) meta for Pearson r on Fisher-z scale ---
def meta_random_effects_pearson(rs, ns):
    """
    rs: array-like of Pearson r per dataset
    ns: array-like of corresponding sample sizes (# of windows used)
    Returns dict with random-effects pooled r, SE, p, 95% CI, k, n_total, tau2, Q, I2.
    """
    rs = np.asarray(rs, dtype=float)
    ns = np.asarray(ns, dtype=float)
    mask = np.isfinite(rs) & np.isfinite(ns) & (ns > 3)
    rs, ns = rs[mask], ns[mask]
    k = rs.size
    if k == 0:
        return dict(r_random=np.nan, z=np.nan, se_z=np.nan, z_stat=np.nan, p=np.nan,
                    ci_low=np.nan, ci_high=np.nan, k=0, n_total=0,
                    tau2=np.nan, Q=np.nan, I2=np.nan)

    # Fisher z and within-study variances
    zs = fisher_z(rs)                                # atanh(r)
    vi = 1.0 / (ns - 3.0)                            # Var(z) ≈ 1/(n-3)
    wi = 1.0 / vi                                    # FE weights

    # Fixed-effect on z (for Q, tau^2)
    z_fe = np.sum(wi * zs) / np.sum(wi)
    Q = np.sum(wi * (zs - z_fe) ** 2)
    df = k - 1
    C = np.sum(wi) - (np.sum(wi**2) / np.sum(wi))
    tau2 = max(0.0, (Q - df) / max(C, 1e-12))        # DerSimonian–Laird

    # Random-effects weights
    wi_star = 1.0 / (vi + tau2)
    z_re = np.sum(wi_star * zs) / np.sum(wi_star)
    se = 1.0 / np.sqrt(np.sum(wi_star))
    z_stat = z_re / se
    p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    # 95% CI on z-scale
    z_lo = z_re - 1.96 * se
    z_hi = z_re + 1.96 * se

    # Back-transform to r
    r_re = inv_fisher_z(z_re)
    ci_low = inv_fisher_z(z_lo)
    ci_high = inv_fisher_z(z_hi)

    # Heterogeneity (I^2)
    I2 = max(0.0, (Q - df) / max(Q, 1e-12)) * 100.0

    return dict(r_random=float(r_re), z=float(z_re), se_z=float(se),
                z_stat=float(z_stat), p=float(p),
                ci_low=float(ci_low), ci_high=float(ci_high),
                k=int(k), n_total=int(np.sum(ns)),
                tau2=float(tau2), Q=float(Q), I2=float(I2))

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Multi-dataset ECG/RR SampEn with meta-analysis.")
    ap.add_argument("inputs", nargs="+", help="CSV files or globs (e.g., data/*.csv)")
    ap.add_argument("--fs", type=float, default=2000.0, help="Sampling rate Hz (default 2000)")
    ap.add_argument("--win-sec", type=float, default=10.0, help="Window length in seconds (default 10)")
    ap.add_argument("--overlap", type=float, default=0.0, help="Window overlap fraction in [0,1) (default 0)")
    ap.add_argument("--ds-ecg", type=int, default=20, help="Downsample factor for ECG SampEn only (default 20)")
    ap.add_argument("--m", type=int, default=2, help="SampEn embedding dimension m (default 2)")
    ap.add_argument("--r-factor", type=float, default=0.2, help="r as fraction of SD (default 0.2)")
    return ap.parse_args()

def expand_inputs(patterns):
    paths = []
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            paths.extend(matches)
        else:
            paths.append(p)
    # unique, preserve order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return [Path(p) for p in out]

def main():
    args = parse_args()
    csv_paths = expand_inputs(args.inputs)
    if len(csv_paths) == 0:
        log("No input files found.")
        return

    per_dataset_stats = []
    for p in csv_paths:
        try:
            stats = analyze_dataset(
                csv_path=p,
                fs_hz=args.fs,
                window_sec=args.win_sec,
                overlap=args.overlap,
                ds_ecg=args.ds_ecg,
                m_sampen=args.m,
                r_factor=args.r_factor
            )
            per_dataset_stats.append(stats)
        except Exception as e:
            log(f"ERROR processing {p}: {e}")

    if len(per_dataset_stats) == 0:
        log("No datasets processed successfully.")
        return

    df_sum = pd.DataFrame(per_dataset_stats)

    # Meta-analysis for Pearson r (ECG and RR separately)
    # Meta-analysis for Pearson r (ECG and RR)
    meta_ecg_fe = meta_fixed_effect_pearson(df_sum["r_ecg"].values, df_sum["n_ecg"].values)
    meta_rr_fe  = meta_fixed_effect_pearson(df_sum["r_rr"].values,  df_sum["n_rr"].values)

    meta_ecg_re = meta_random_effects_pearson(df_sum["r_ecg"].values, df_sum["n_ecg"].values)
    meta_rr_re  = meta_random_effects_pearson(df_sum["r_rr"].values,  df_sum["n_rr"].values)

    log("\n=== Meta-analysis (Pearson r) ===")
    log(f"[Fixed]  ECG SampEn vs Temp: r = {meta_ecg_fe['r_fixed']:.3f} "
        f"(95% CI {meta_ecg_fe['ci_low']:.3f}..{meta_ecg_fe['ci_high']:.3f}), "
        f"p = {meta_ecg_fe['p']:.3g} | k={meta_ecg_fe['k']}, n_total={meta_ecg_fe['n_total']}")
    log(f"[Random] ECG SampEn vs Temp: r = {meta_ecg_re['r_random']:.3f} "
        f"(95% CI {meta_ecg_re['ci_low']:.3f}..{meta_ecg_re['ci_high']:.3f}), "
        f"p = {meta_ecg_re['p']:.3g} | k={meta_ecg_re['k']}, n_total={meta_ecg_re['n_total']}, "
        f"tau²={meta_ecg_re['tau2']:.4f}, Q={meta_ecg_re['Q']:.3f}, I²={meta_ecg_re['I2']:.1f}%")

    log(f"[Fixed]  RR  SampEn vs Temp: r = {meta_rr_fe['r_fixed']:.3f} "
        f"(95% CI {meta_rr_fe['ci_low']:.3f}..{meta_rr_fe['ci_high']:.3f}), "
        f"p = {meta_rr_fe['p']:.3g} | k={meta_rr_fe['k']}, n_total={meta_rr_fe['n_total']}")
    log(f"[Random] RR  SampEn vs Temp: r = {meta_rr_re['r_random']:.3f} "
        f"(95% CI {meta_rr_re['ci_low']:.3f}..{meta_rr_re['ci_high']:.3f}), "
        f"p = {meta_rr_re['p']:.3g} | k={meta_rr_re['k']}, n_total={meta_rr_re['n_total']}, "
        f"tau²={meta_rr_re['tau2']:.4f}, Q={meta_rr_re['Q']:.3f}, I²={meta_rr_re['I2']:.1f}%")


    # Save summary
    out_summary = df_sum.copy()
    # Fixed-effect
    for prefix, meta in [("ecg_fe", meta_ecg_fe), ("rr_fe", meta_rr_fe)]:
        for k in ["r_fixed", "ci_low", "ci_high", "p", "k", "n_total"]:
            out_summary[f"meta_{prefix}_{k}"] = meta[k]
    # Random-effects
    for prefix, meta in [("ecg_re", meta_ecg_re), ("rr_re", meta_rr_re)]:
        for k in ["r_random", "ci_low", "ci_high", "p", "k", "n_total", "tau2", "Q", "I2", "se_z"]:
            out_summary[f"meta_{prefix}_{k}"] = meta.get(k, np.nan)

    out_summary.to_csv("summary_meta_analysis.csv", index=False)
    log("\nSaved: summary_meta_analysis.csv")

if __name__ == "__main__":
    main()
