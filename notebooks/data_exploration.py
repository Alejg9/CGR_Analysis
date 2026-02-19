# eda_cgr.py (versión mejorada)
import os
import warnings
warnings.filterwarnings("ignore")

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Config ---
INPUT_FILE = "data/CGR.xlsx"
DATA_COLUMNS = [
    'tmax(degC)',
    'tmin(degC)',
    'ppt(mm)',
    'ws(mps)',
    'aet(mm)',
    'q(mm)',
    'soil(mm)'
]
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funciones auxiliares ---
def to_datetime_from_yyyymm(series):
    s = series.astype(str)
    return pd.to_datetime(s + "01", format="%Y%m%d")

def basic_stats(arr):
    arr = np.asarray(arr[~np.isnan(arr)])
    if arr.size == 0:
        return {}
    return {
        "count": arr.size,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": float(np.percentile(arr,75) - np.percentile(arr,25)),
        "skew": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr))
    }

def detect_outliers_iqr(arr):
    arr = np.asarray(arr[~np.isnan(arr)])
    if arr.size == 0:
        return np.array([], dtype=int)
    q1, q3 = np.percentile(arr, [25,75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return np.where((arr < low) | (arr > high))[0]

def detect_outliers_zscore(arr, thresh=3.0):
    arr = np.asarray(arr[~np.isnan(arr)])
    if arr.size == 0:
        return np.array([], dtype=int)
    z = np.abs((arr - np.mean(arr)) / np.std(arr, ddof=1))
    return np.where(z > thresh)[0]

# --- Distribuciones a probar ---
DIST_NAMES_GENERAL = ["norm", "t"]
DIST_NAMES_POSITIVE = ["lognorm", "gamma", "weibull_min", "expon"]

def fit_distribution_and_tests(data, dist_name, bins=20):
    """Ajusta la distribución y realiza KS y Chi-cuadrado"""
    data = np.asarray(data[~np.isnan(data)])
    if data.size < 10:
        return None

    # Seleccionar subconjunto si la dist. requiere positivos
    if dist_name in DIST_NAMES_POSITIVE:
        data = data[data > 0]
        if data.size < 10:
            return None

    try:
        if dist_name == "norm":
            params = stats.norm.fit(data)
            cdf_func = lambda x: stats.norm.cdf(x, *params)
            pdf_func = lambda x: stats.norm.pdf(x, *params)
            ks = stats.kstest(data, "norm", args=params)

        elif dist_name == "t":
            params = stats.t.fit(data)
            cdf_func = lambda x: stats.t.cdf(x, *params)
            pdf_func = lambda x: stats.t.pdf(x, *params)
            ks = stats.kstest(data, lambda x: stats.t.cdf(x, *params))

        elif dist_name == "lognorm":
            params = stats.lognorm.fit(data, floc=0)
            cdf_func = lambda x: stats.lognorm.cdf(x, *params)
            pdf_func = lambda x: stats.lognorm.pdf(x, *params)
            ks = stats.kstest(data, lambda x: stats.lognorm.cdf(x, *params))

        elif dist_name == "gamma":
            params = stats.gamma.fit(data, floc=0)
            cdf_func = lambda x: stats.gamma.cdf(x, *params)
            pdf_func = lambda x: stats.gamma.pdf(x, *params)
            ks = stats.kstest(data, lambda x: stats.gamma.cdf(x, *params))

        elif dist_name == "weibull_min":
            params = stats.weibull_min.fit(data, floc=0)
            cdf_func = lambda x: stats.weibull_min.cdf(x, *params)
            pdf_func = lambda x: stats.weibull_min.pdf(x, *params)
            ks = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, *params))

        elif dist_name == "expon":
            params = stats.expon.fit(data, floc=0)
            cdf_func = lambda x: stats.expon.cdf(x, *params)
            pdf_func = lambda x: stats.expon.pdf(x, *params)
            ks = stats.kstest(data, lambda x: stats.expon.cdf(x, *params))

        else:
            return None

        # --- Chi-cuadrado ---
        counts, bin_edges = np.histogram(data, bins=bins)
        expected_probs = np.diff([cdf_func(be) for be in bin_edges])
        expected_counts = expected_probs * data.size
        mask = expected_counts > 0
        counts, expected_counts = counts[mask], expected_counts[mask]
        chi2_stat, chi2_p = stats.chisquare(
            f_obs=counts, f_exp=expected_counts, ddof=len(params)
        )

        return {
            "params": tuple(np.round(params, 6)),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "chi2_stat": float(chi2_stat),
            "chi2_pvalue": float(chi2_p),
            "cdf_func": cdf_func,
            "pdf_func": pdf_func
        }

    except Exception as e:

        print(f"[ERROR] Distribución {dist_name} falló: {e}")

        return None

# --- Gráficos ---
def plot_timeseries_and_decomposition(series, name, outdir):
    fig, ax = plt.subplots(2,1, figsize=(12,8), gridspec_kw={"height_ratios":[2,1]})
    series.plot(ax=ax[0], title=f"{name} - time series")
    series.rolling(window=12, min_periods=1).mean().plot(ax=ax[0], label="MA(12)")
    ax[0].legend()
    ax[0].set_ylabel(name)

    try:
        decomp = seasonal_decompose(series.dropna(), period=12, model="additive")
        decomp.trend.plot(ax=ax[1], label="Trend")
        ax[1].legend()
    except:
        ax[1].text(0.5,0.5,"Decomposition failed",ha="center")
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_timeseries_decomp.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_histogram_with_fits(sample, name, fit_results, outdir):
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    n, bins, _ = ax[0].hist(sample, bins=30, alpha=0.6)
    x = np.linspace(np.nanmin(sample), np.nanmax(sample), 300)
    ax[0].set_title(f"{name} - Histogram + PDFs")
    for dist_name, res in fit_results.items():
        if res is None: continue
        pdf_vals = res['pdf_func'](x)
        bin_w = bins[1]-bins[0]
        ax[0].plot(x, pdf_vals*len(sample)*bin_w, label=dist_name)
    ax[0].legend()

    # CDF
    sorted_s = np.sort(sample)
    emp_cdf = np.arange(1, len(sorted_s)+1)/len(sorted_s)
    ax[1].step(sorted_s, emp_cdf, where='post', label='Empirical')
    for dist_name, res in fit_results.items():
        if res is None: continue
        theo_cdf = res['cdf_func'](sorted_s)
        ax[1].plot(sorted_s, theo_cdf, label=f"{dist_name}")
    ax[1].legend()
    ax[1].set_title("Empirical vs Theoretical CDF")
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_hist_cdf.png")
    plt.savefig(path)
    plt.close()
    return path

# --- Pipeline principal ---
def analyze(raw_df, data_columns, output_dir):
    try:
        pandas_dates = to_datetime_from_yyyymm(raw_df["Date"].to_numpy())
    except:
        pandas_dates = pd.to_datetime(raw_df["Date"].to_pandas(), errors="coerce")

    pdf = raw_df.to_pandas().copy()
    pdf.index = pd.DatetimeIndex(pandas_dates)
    results_summary = {}

    for col in data_columns:
        if col not in pdf.columns:
            continue
        series = pdf[col].astype(float)
        name = col.replace("(", "").replace(")", "").replace("/", "_")

        stats_info = basic_stats(series.values)
        out_iqr = detect_outliers_iqr(series.values)
        out_z = detect_outliers_zscore(series.values)

        # Decide tipo de distribuciones
        if (series <= 0).sum() > 0:
            dists = DIST_NAMES_GENERAL
        else:
            dists = DIST_NAMES_GENERAL + DIST_NAMES_POSITIVE

        fit_results = {}
        for d in dists:
            fit_results[d] = fit_distribution_and_tests(series.values, d)
        print(fit_results)
        # Gráficos
        ts_plot = plot_timeseries_and_decomposition(series, name, output_dir)
        hist_plot = plot_histogram_with_fits(series.dropna().values, name, fit_results, output_dir)

        # Armar resumen
        fit_summary = {}
        for d, r in fit_results.items():
            if r is not None:
                fit_summary[d] = {
                    "params": r["params"],
                    "ks_pvalue": r["ks_pvalue"],
                    "chi2_pvalue": r["chi2_pvalue"]
                }

        results_summary[col] = {
            "basic_stats": stats_info,
            "outliers_iqr_count": len(out_iqr),
            "outliers_z_count": len(out_z),
            "fits": fit_summary,
            "timeseries_plot": ts_plot,
            "hist_cdf_plot": hist_plot
        }

    import json
    with open(os.path.join(output_dir, "eda_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ EDA completado. Resultados en: {output_dir}")
    return results_summary

if __name__ == "__main__":
    raw = pl.read_excel(INPUT_FILE)
    summary = analyze(raw, DATA_COLUMNS, OUTPUT_DIR)
