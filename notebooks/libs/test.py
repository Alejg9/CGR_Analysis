import numpy as np
from scipy import stats
import polars as pl

def test_distributions(series: pl.Series, name):
    data = np.asarray(series.drop_nulls())
    print(f"\n=== {name} ===")
    print(f"Count: {len(data)}, Min: {data.min()}, Max: {data.max()}")

    distributions = ["norm", "t", "lognorm", "gamma", "weibull_min", "expon"]
    for dist_name in distributions:
        try:
            # Para las positivas, quitamos los <=0
            if dist_name in ["lognorm", "gamma", "weibull_min", "expon"]:
                data = data[data > 0]
                if len(data) < 10:
                    print(f"  ⚠️ {dist_name} omitida: pocos datos > 0")
                    continue

            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            ks = stats.kstest(data, dist_name, args=params)
            print(f"  ✅ {dist_name}: KS p={ks.pvalue:.4f}, stat={ks.statistic:.4f}, params={np.round(params, 4)}")
        except Exception as e:
            print(f"  ❌ {dist_name} falló: {e}")


# --- Prueba con tus columnas ---

if __name__ == "__main__":
    raw_df = pl.read_excel("CGR1.xlsx")
    for col in [
        'tmax(degC)',
        'tmin(degC)',
        'ppt(mm)',
        'ws(mps)',
        'aet(mm)',
        'q(mm)',
        'soil(mm)'
    ]:
        test_distributions(raw_df[col], col)
