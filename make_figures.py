# make_figures.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_windows_csvs(paths):
    items = []
    for p in paths:
        df = pd.read_csv(p)
        df["_dataset"] = Path(p).stem.replace("__windowed_sampen_and_temp", "")
        items.append(df)
    return items

def scatter_per_dataset(dfs):
    for df in dfs:
        name = df["_dataset"].iloc[0]
        # ECG SampEn scatter
        plt.figure(figsize=(5,5))
        m = np.isfinite(df["temp_win_mean"]) & np.isfinite(df["sampen_ecg_ds"])
        plt.scatter(df.loc[m,"temp_win_mean"], df.loc[m,"sampen_ecg_ds"], s=15)
        plt.xlabel("Hand temperature (°C)")
        plt.ylabel("SampEn (ECG, downsampled)")
        plt.title(f"{name}: Temp vs ECG SampEn")
        plt.tight_layout()
        plt.show()

        # RR SampEn scatter
        plt.figure(figsize=(5,5))
        m = np.isfinite(df["temp_win_mean"]) & np.isfinite(df["sampen_rr"])
        plt.scatter(df.loc[m,"temp_win_mean"], df.loc[m,"sampen_rr"], s=15)
        plt.xlabel("Hand temperature (°C)")
        plt.ylabel("SampEn (RR intervals)")
        plt.title(f"{name}: Temp vs RR SampEn")
        plt.tight_layout()
        plt.show()

def timeseries_per_dataset(dfs):
    for df in dfs:
        name = df["_dataset"].iloc[0]
        plt.figure(figsize=(10,4))
        plt.plot(df["time_center_s"], df["temp_win_mean"], label="Hand Temp")
        plt.plot(df["time_center_s"], df["sampen_ecg_ds"], label="ECG SampEn (ds)")
        plt.plot(df["time_center_s"], df["sampen_rr"], label="RR SampEn")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"{name}: Windowed Temp & SampEn")
        plt.legend()
        plt.tight_layout()
        plt.show()

def forest_from_summary(summary_csv, which="ecg"):
    """
    which: 'ecg' or 'rr'
    Expects columns from your summary script:
      - per-dataset r: r_ecg / r_rr
      - FE CI: meta_ecg_fe_ci_low/high or meta_rr_fe_ci_low/high
      - pooled FE r & p: meta_ecg_fe_* / meta_rr_fe_*
      - pooled RE (optional): meta_ecg_re_* / meta_rr_re_*
    """
    df = pd.read_csv(summary_csv)

    label_col = "path" if "path" in df.columns else None
    labels = [Path(p).stem if label_col else f"ds{i+1}" for i, p in enumerate(df[label_col])] if label_col else [f"ds{i+1}" for i in range(len(df))]
    r_col = f"r_{which}"
    r_vals = df[r_col].to_numpy()

    # Prefer dataset-specific CIs if present; otherwise compute Wald ~95% CI for r via Fisher z per dataset.
    ci_lo_list, ci_hi_list = [], []
    for i in range(len(df)):
        # try FE dataset-level CIs if your summary wrote them per-dataset (often not); otherwise compute quick CI
        r = r_vals[i]
        n_col = f"n_{which}"
        n = df[n_col].iloc[i] if n_col in df.columns else np.nan
        if np.isfinite(r) and np.isfinite(n) and n > 3:
            z = np.arctanh(np.clip(r, -0.999999, 0.999999))
            se = 1.0/np.sqrt(n-3.0)
            zlo, zhi = z - 1.96*se, z + 1.96*se
            ci_lo_list.append(np.tanh(zlo))
            ci_hi_list.append(np.tanh(zhi))
        else:
            ci_lo_list.append(np.nan)
            ci_hi_list.append(np.nan)

    # Pooled (fixed)
    fe_prefix = f"meta_{which}_fe_"
    fe_r = df.get(fe_prefix + "r_fixed", pd.Series([np.nan])).iloc[0]
    fe_lo = df.get(fe_prefix + "ci_low", pd.Series([np.nan])).iloc[0]
    fe_hi = df.get(fe_prefix + "ci_high", pd.Series([np.nan])).iloc[0]
    fe_p  = df.get(fe_prefix + "p", pd.Series([np.nan])).iloc[0]

    # Pooled (random) optional
    re_prefix = f"meta_{which}_re_"
    re_r  = df.get(re_prefix + "r_random", pd.Series([np.nan])).iloc[0]
    re_lo = df.get(re_prefix + "ci_low", pd.Series([np.nan])).iloc[0]
    re_hi = df.get(re_prefix + "ci_high", pd.Series([np.nan])).iloc[0]
    re_p  = df.get(re_prefix + "p", pd.Series([np.nan])).iloc[0]
    re_I2 = df.get(re_prefix + "I2", pd.Series([np.nan])).iloc[0]

    # Plot
    plt.figure(figsize=(7, max(3, 0.6*len(labels) + 2)))
    y = np.arange(len(labels))
    for i, (r, lo, hi) in enumerate(zip(r_vals, ci_lo_list, ci_hi_list)):
        if np.isfinite(r):
            if np.isfinite(lo) and np.isfinite(hi):
                plt.plot([lo, hi], [i, i], "-", lw=1)
            plt.plot(r, i, "o")
    plt.axvline(0, ls="--", lw=1)
    plt.yticks(y, labels)
    title_metric = "ECG SampEn" if which=="ecg" else "RR SampEn"
    plt.xlabel(f"Pearson r ({title_metric} vs Temp)")
    title = f"Forest plot ({title_metric})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Add a pooled figure summarizing FE (+ RE if available)
    plt.figure(figsize=(6,4))
    xs = []
    labs = []
    los = []
    his = []
    if np.isfinite(fe_r):
        xs.append(fe_r); labs.append("Fixed-effect"); los.append(fe_lo); his.append(fe_hi)
    if np.isfinite(re_r):
        xs.append(re_r); labs.append("Random-effects"); los.append(re_lo); his.append(re_hi)
    if xs:
        yy = np.arange(len(xs))
        for i, (x, lo, hi) in enumerate(zip(xs, los, his)):
            if np.isfinite(lo) and np.isfinite(hi):
                plt.plot([lo, hi], [i, i], "-", lw=2)
            plt.plot(x, i, "o")
        plt.axvline(0, ls="--", lw=1)
        plt.yticks(yy, labs)
        xlab = f"Pooled r ({title_metric} vs Temp)"
        # annotate p-values
        txt = []
        if np.isfinite(fe_r):
            txt.append(f"FE p={fe_p:.3g}")
        if np.isfinite(re_r):
            txt.append(f"RE p={re_p:.3g}, I²={re_I2:.1f}%")
        plt.xlabel(xlab + ("  [" + " | ".join(txt) + "]" if txt else ""))
        plt.title(f"Pooled effects ({title_metric})")
        plt.tight_layout()
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Make figures from windowed CSVs and meta-analysis summary.")
    ap.add_argument("--windows", nargs="+", required=True,
                    help="Per-dataset window CSVs (e.g., dataset1__windowed_sampen_and_temp.csv ...)")
    ap.add_argument("--summary", required=True,
                    help="summary_meta_analysis.csv produced by the multi-dataset script")
    args = ap.parse_args()

    dfs = read_windows_csvs(args.windows)

    # 1) Scatterplots per dataset (Temp vs SampEn)
    scatter_per_dataset(dfs)

    # 2) Time-series overlays per dataset
    timeseries_per_dataset(dfs)

    # 3) Forest plots (ECG then RR)
    forest_from_summary(args.summary, which="ecg")
    forest_from_summary(args.summary, which="rr")

    # Optional: save a single PDF with all figures (uncomment to enable)
    # from matplotlib.backends.backend_pdf import PdfPages
    # with PdfPages("figures_all_panels.pdf") as pdf:
    #     # Re-run the plotting functions but call pdf.savefig() after each figure instead of plt.show()
    #     pass

if __name__ == "__main__":
    main()
