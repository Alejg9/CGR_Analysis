import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
import polars as pl


class DataAnalysisEngine:
    def __init__(self, input_file, data_columns, output_dir):
        """
        Initialize the DataAnalysisEngine with input file, data columns, and output directory.

        :param input_file: Path to the Excel input file
        :param data_columns: List of columns to analyze
        :param output_dir: Directory to save results and plots
        """
        self.input_file = input_file
        self.data_columns = data_columns
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _convert_to_datetime(self, series):
        """
        Convert a series of date strings to datetime objects.

        :param series: Input pandas Series containing date strings
        :return: Series with datetime values
        """
        return pd.to_datetime(series, format='%Y%m%d')

    def _calculate_basic_stats(self, series):
        """
        Calculate basic statistics for a series (mean, standard deviation).

        :param series: Input pandas Series
        :return: Dictionary containing stats
        """
        return {'mean': series.mean(), 'std': series.std()}

    def _detect_outliers_iqr(self, series):
        """
        Detect outliers using the Interquartile Range (IQR) method.

        :param series: Input pandas Series
        :return: Series of outliers
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return series[(series < lower_bound) | (series > upper_bound)].tolist()

    def _detect_outliers_zscore(self, series: pd.Series):
        """
        Detect outliers using Z-score method.

        :param series: Input pandas Series
        :return: Series of outliers
        """
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std
        return series[np.abs(z_scores) > 3].tolist()

    def _fit_distribution(self, series):
        """
        Fit a normal distribution to the data.

        :param series: Input pandas Series
        :return: Dictionary containing distribution parameters and fit
        """
        params = norm.fit(series)
        return {'params': params, 'fit': norm.pdf(series, *params).tolist()}

    def _generate_timeseries_plot(self, series, name, output_dir):
        """
        Generate and save a time series plot.

        :param series: Input pandas Series with datetime index
        :param name: Column name for the plot
        :param output_dir: Directory to save the plot
        :return: Path to saved plot
        """
        plt.figure(figsize=(10, 5))
        plt.plot(series)
        plt.title(f'Time Series: {name}')
        plt.savefig(os.path.join(output_dir, f'{name}_timeseries.png'))
        plt.close()
        return os.path.join(output_dir, f'{name}_timeseries.png')

    def _generate_decomposition_plot(self, series, name, output_dir, date):
        """
        Generate and save a time series decomposition plot.

        :param series: Input pandas Series with datetime index
        :param name: Column name for the plot
        :param output_dir: Directory to save the plot
        :return: Path to saved plot
        """
        decomposed = seasonal_decompose(series,period=12, model='additive', extrapolate_trend=2)
        plt.figure(figsize=(10, 5))
        decomposed.plot()
        plt.savefig(os.path.join(output_dir, f'{name}_decomposition.png'))
        plt.close()

        data = {
            'date': date,
            'seasonal': decomposed.seasonal,
            'trend': decomposed.trend,
            'resid': decomposed.resid,
            'observed': decomposed.observed,
            'weights': decomposed.weights
        }

        print(data)

        df = pl.DataFrame(data)
        df = df.with_columns([
            (pl.col('trend') + pl.col('resid')).alias('rt'),
        ])
        df.write_csv(os.path.join(output_dir, f'{name}_decomposition.csv'))
        return os.path.join(output_dir, f'{name}_decomposition.png')

    def _generate_histogram_plot(self, series, name, output_dir):
        """
        Generate and save a histogram with distribution fit.

        :param series: Input pandas Series
        :param name: Column name for the plot
        :param output_dir: Directory to save the plot
        :return: Path to saved plot
        """
        plt.figure(figsize=(10, 5))
        plt.hist(series, bins=50, alpha=0.7)
        mean, std = norm.fit(series)
        x = np.linspace(min(series), max(series), 100)
        plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2)
        plt.title(f'Histogram: {name} with Normal Fit')
        plt.savefig(os.path.join(output_dir, f'{name}_histogram.png'))
        plt.close()
        return os.path.join(output_dir, f'{name}_histogram.png')

    def analyze(self):
        """
        Execute the full analysis pipeline for all columns.

        :return: List of analysis results per column
        """
        raw = pd.read_excel(self.input_file)
        results = []

        date = raw['Fecha']
        for col in self.data_columns:
            series = raw[col]

            # Convert date column to datetime if necessary
            if col == 'date':
                series = self._convert_to_datetime(series)

            # Calculate basic stats
            stats = self._calculate_basic_stats(series)

            # Detect outliers using IQR method
            iqr_outliers = self._detect_outliers_iqr(series)

            # Detect outliers using Z-score method
            zscore_outliers = self._detect_outliers_zscore(series)

            # Fit a normal distribution
            distribution_fit = self._fit_distribution(series)

            # Generate time series plot
            ts_plot = self._generate_timeseries_plot(series, col, self.output_dir)

            # Generate decomposition plot
            decomposition_plot = self._generate_decomposition_plot(series, col, self.output_dir, date=date)

            # Generate histogram plot
            histogram_plot = self._generate_histogram_plot(series, col, self.output_dir)

            # Store results
            results.append({
                'column': col,
                'stats': stats,
                'iqr_outliers': iqr_outliers,
                'zscore_outliers': zscore_outliers,
                'distribution_fit': distribution_fit,
                'time_series_plot': ts_plot,
                'decomposition_plot': decomposition_plot,
                'histogram_plot': histogram_plot
            })

        print(results)
        results = {
            'data': results,
        }
        print(results)
        # Save results to JSON
        with open(os.path.join(self.output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f)

        return results


