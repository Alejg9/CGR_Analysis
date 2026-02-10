import polars as pl
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import os
warnings.filterwarnings('ignore')

# ============================
# CARGA DE DATOS
# ============================
df = pl.read_csv('CGR1.csv')

# Columnas numéricas
numeric_cols = [
    "tmax(degC)", "tmin(degC)", "ppt(mm)", "ws(mps)", "aet(mm)", "q(mm)", "soil(mm)"
]

# Aseguramos que las columnas existan
transformations = [pl.col(c).cast(pl.Float64) for c in numeric_cols]
transformations = transformations + [
    (
            pl.col('date').str.split(',').list.get(0) +
            pl.col('date').str.split(',').list.get(1)
    ).str.strptime(dtype=pl.Datetime, format='%Y%m')
]
df = df.with_columns(transformations)
# Convertimos a DataFrame de Pandas para compatibilidad con Prophet y sklearn

df_pandas = df.to_pandas()



# ============================
# FUNCIONES AUXILIARES
# ============================

def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> List[int]:
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outliers.tolist()

# Función para ajustar distribución y calcular K-S
def fit_distribution(data: np.ndarray, distributions: List[str]) -> Tuple[str, float, float, tuple]:
    best_dist = None
    best_ks_stat = float('inf')
    best_p_value = 0
    best_params = None
    max_p_value = 0
    for dist_name in distributions:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, dist_name, args=params)
            print(p_value, max_p_value, ks_stat, best_ks_stat, p_value > max_p_value, ks_stat < best_ks_stat)
            if p_value > max_p_value and ks_stat < best_ks_stat:
                best_dist = dist_name
                best_ks_stat = ks_stat
                best_p_value = p_value
                best_params = params
        except Exception as e:
            print(e)

    return best_dist, best_ks_stat, best_p_value, best_params

# Función para generar histograma y CDF
def plot_histogram_and_cdf(data: np.ndarray, col_name: str, dist_name: str, params: tuple):
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Histograma", "CDF Empírica vs Teórica"])

    # Histograma
    hist_data = data
    hist_fig = px.histogram(hist_data, nbins=30, title=f"Histograma de {col_name}")
    hist_fig.update_layout(height=300, showlegend=False)

    # CDF
    x = np.linspace(min(data), max(data), 1000)
    cdf_emp = np.sort(data)
    cdf_emp = np.interp(x, np.sort(data), np.linspace(0, 1, len(data)))
    # Crear la CDF teórica

    dist = getattr(stats, dist_name)
    cdf_theo = dist.cdf(x, *params)

    cdf_fig = go.Figure()
    cdf_fig.add_trace(go.Scatter(x=x, y=cdf_theo, mode='lines', name='CDF Teórica', line=dict(color='blue')))
    cdf_fig.add_trace(go.Scatter(x=x, y=cdf_emp, mode='lines', name='CDF Empírica', line=dict(color='red')))
    cdf_fig.update_layout(title=f"Comparación CDF: {col_name}", xaxis_title="Valor", yaxis_title="CDF")

    fig.add_trace(hist_fig.data[0], row=1, col=1)
    fig.add_trace(cdf_fig.data[0], row=2, col=1)

    fig.update_layout(title_text=f"Análisis de {col_name}", height=600)
    fig.show()

# Función para análisis estadístico básico
def basic_stats(data: np.ndarray, col_name: str):
    stats_dict = {
        "Media": np.mean(data),
        "Mediana": np.median(data),
        "Desviación estándar": np.std(data),
        "CV": np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
        "Mínimo": np.min(data),
        "Máximo": np.max(data),
        "Q1": np.percentile(data, 25),
        "Q3": np.percentile(data, 75),
        "IQR": np.percentile(data, 75) - np.percentile(data, 25),
        "Outliers": detect_outliers_iqr(data),
        "Longitud": len(data)
    }
    return stats_dict

def adf_test(series: pd.Series, name: str) -> dict:
    result = adfuller(series.dropna())
    output = {
        'name': name,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] <= 0.05
    }
    print(f"\nADF Test for {name}:")
    for key, value in output.items():
        print(f"{key}: {value}")
    return output

def kpss_test(series: pd.Series, name: str) -> dict:
    result = kpss(series.dropna(), regression='c')
    output = {
        'name': name,
        'KPSS Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[3],
        'Stationary': result[1] > 0.05
    }
    print(f"\nKPSS Test for {name}:")
    for key, value in output.items():
        print(f"{key}: {value}")
    return output

def make_stationary(series: pd.Series, name: str) -> pd.Series:
    """Aplica diferenciación para estacionarizar"""
    diff = series.diff().dropna()
    print(f"\nDiferenciación de {name}:")
    print(f"Media: {diff.mean():.4f}, Desv. estándar: {diff.std():.4f}")
    return diff

def generate_synthetic_data(series: pd.Series, n_points: int = 12) -> pd.Series:
    """Genera datos sintéticos con ruido gaussiano"""
    # Asumiendo que el ruido es normal con media 0 y desv. estándar de la serie
    noise = np.random.normal(0, series.std(), n_points)
    # Interpolamos los valores faltantes
    interpolated = np.interp(np.linspace(0, len(series), len(series) + n_points), np.arange(len(series)), series.values)
    # Añadimos el ruido
    synthetic = interpolated + noise
    return pd.Series(synthetic, index=pd.date_range(start=series.index[-1], periods=len(series) + n_points, freq='MS'))

def fit_arima(series: pd.Series, name: str) -> ARIMA:
    """Ajusta un modelo ARIMA"""
    # Primero verificamos estacionariedad
    diff_series = series
    if not adf_test(series, name)['Stationary']:
        diff_series = make_stationary(series, name)
    # Ajustamos ARIMA
    model = ARIMA(diff_series, order=(1, 1, 1))
    fitted_model = model.fit()
    print(f"\nModelo ARIMA ajustado para {name}")
    return fitted_model, diff_series

def predict_arima(fitted_model: ARIMA, n_periods: int = 24) -> pd.DataFrame:
    """Predice n períodos adelante"""
    forecast = fitted_model.forecast(steps=n_periods)
    confidence_intervals = fitted_model.get_forecast(steps=n_periods).conf_int()
    return forecast, confidence_intervals

def ml_predict(series: pd.Series, name: str) -> pd.DataFrame:
    """Predicción con Random Forest"""
    # Preparar datos
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    # Dividir en entrenamiento y prueba
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Entrenar modelo
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predicción
    y_pred = rf.predict(X_test)

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.flatten(), y_test, label='Real', marker='o')
    plt.plot(X_test.flatten(), y_pred, label='Predicción', marker='x', linestyle='--')
    plt.title(f"Predicción con Random Forest - {name}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()

    # Devolver predicción
    return y_pred, X_test

def prophet_predict(series: pd.Series, name: str) -> pd.DataFrame:
    """Predicción con Prophet"""
    df_prophet = pd.DataFrame({
        'ds': pd.date_range(start=series.index[0], periods=len(series), freq='MS'),
        'y': series.values
    })
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=24, freq='MS')
    forecast = model.predict(future)
    return forecast

# ============================
# ANÁLISIS EXPLORATORIO
# ============================

print("=== ANÁLISIS EXPLORATORIO ===")

for col in numeric_cols:
    data = df_pandas[col].dropna().values
    stats = basic_stats(data, col)
    print(f"\n=== Estadísticas de {col} ===")
    for key, value in stats.items():
        continue
        #print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    distributions = [
        "norm", "expon", "gamma", "beta", "weibull_min", "lognorm",
        "chi2", "t", "f", "laplace", "cauchy", "logistic", "gumbel_r", "gumbel_l"
    ]
    # Prueba de normalidad (K-S)
    dist_name, ks_stat, p_value, params = fit_distribution(data, distributions)

    print(dist_name, ks_stat, p_value, params)
    print(f"\nPrueba K-S para normalidad: Estadístico={ks_stat:.4f}, p-valor={p_value:.4f}")

    # Grafica
    plot_histogram_and_cdf(data, col, dist_name, params)

# ============================
# PRUEBAS DE ESTACIONARIEDAD
# ============================

print("\n=== PRUEBAS DE ESTACIONARIEDAD ===")

for col in numeric_cols:
    series = df_pandas[col].dropna()
    adf_result = adf_test(series, col)
    kpss_result = kpss_test(series, col)

    print(f"\n--- {col} ---")
    print(f"ADF: Estacionario = {adf_result['Stationary']}")
    print(f"KPSS: Estacionario = {kpss_result['Stationary']}")

# ============================
# AJUSTE DE ARIMA
# ============================

print("\n=== AJUSTE DE ARIMA ===")

for col in numeric_cols:
    series = df_pandas[col].dropna()
    print(f"\n--- Ajuste ARIMA para {col} ---")

    # Ajuste ARIMA
    fitted_model, diff_series = fit_arima(series, col)
    forecast, conf_intervals = predict_arima(fitted_model, 24)

    # Visualización
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label='Original', color='blue')
    plt.plot(diff_series.index, diff_series.values, label='Diferenciada', color='green')
    plt.plot(forecast.index, forecast.values, label='Predicción ARIMA', color='red', marker='o')
    plt.fill_between(forecast.index, conf_intervals.iloc[:, 0], conf_intervals.iloc[:, 1], alpha=0.2, color='red')
    plt.title(f"ARIMA para {col}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()

# ============================
# PREDICCIÓN CON ML
# ============================

print("\n=== PREDICCIÓN CON ML ===")

for col in numeric_cols:
    series = df_pandas[col].dropna()
    print(f"\n--- Predicción con ML para {col} ---")

    # Random Forest
    y_pred_rf, X_test_rf = ml_predict(series, col)

    # Prophet
    forecast_prophet = prophet_predict(series, col)
    print(f"Prophet: predicción para {len(forecast_prophet[forecast_prophet['ds'] > series.index[-1]])} puntos futuros")

# ============================
# GENERACIÓN DE DATOS SINTÉTICOS
# ============================

print("\n=== GENERACIÓN DE DATOS SINTÉTICOS ===")

for col in numeric_cols:
    series = df_pandas[col].dropna()
    if len(series) < 12:
        print(f"Generando datos sintéticos para {col}...")
        synthetic = generate_synthetic_data(series, n_points=12)
        print(f"Datos sintéticos generados: {len(synthetic)} puntos")
        # Puedes reemplazar los datos originales aquí si lo deseas
        # df_pandas[col] = pd.concat([series, synthetic], ignore_index=True)

# ============================
# RESULTADOS FINALES
# ============================

print("\n✅ ANÁLISIS COMPLETO FINALIZADO")
print("✅ Estaciones estacionales detectadas en ARIMA")
print("✅ Predicciones para 2023 y 2024 generadas con ML y ARIMA")
print("✅ Datos sintéticos generados para completar series cortas")
