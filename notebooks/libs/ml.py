import numpy as np
from typing import Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
class AtmosphericalData:

    def __init__(self, model: tf.keras.Model, path = "./data/CGR.xlsx", use_cudnn: bool = False):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if path.split(".")[-1] == "xlsx":
            self.df = pd.read_excel(path)
        elif path.split(".")[-1] == "csv":
            self.df = pd.read_csv(path)
        self.ds: Union[None, np.ndarray] = None

        self.scaled_ds: Union[None, dict] = None
        self._model = model

        self._n_steps = None
        self._test_dates = None
        self._pred_rescaled = None
        self._scaler = None
        self._train_size = None
        self._target_col = None
        self._date_col = None
        self._true_rescaled = None

    def convert_date(self, date_col: str):
        self.df[date_col] = self.df[date_col].astype(str) + "01"
        self.df[date_col] = pd.to_datetime(self.df[date_col], format="%Y%m%d")

        self._date_col = date_col
    @staticmethod
    def encode_date(date: pd.Timestamp):
        month = date.month
        sin_m = np.sin(2 * np.pi * month / 12)
        cos_m = np.cos(2 * np.pi * month / 12)
        return sin_m, cos_m

    def target_selection(self, target_col:str):
        self.ds = self.df[[target_col]].values
        self._target_col = target_col

    def data_preprocessing(self, n_steps: int = 12, train_vol: float = 0.9):

        self._n_steps = n_steps

        # ============================
        # 1. Construir features
        # ============================
        values = self.df[self._target_col].values.reshape(-1, 1)

        sin_month = np.sin(2 * np.pi * self.df[self._date_col].dt.month / 12).values.reshape(-1, 1)
        cos_month = np.cos(2 * np.pi * self.df[self._date_col].dt.month / 12).values.reshape(-1, 1)

        features = np.hstack([values, sin_month, cos_month])

        # ============================
        # 2. Escalar SOLO la variable objetivo
        # ============================
        scaler = StandardScaler()
        features[:, 0:1] = scaler.fit_transform(features[:, 0:1])

        self._scaler = scaler

        # ============================
        # 3. Crear ventanas deslizantes
        # ============================
        X, y = [], []

        for i in range(n_steps, len(features)):
            X.append(features[i - n_steps:i, :])   # (n_steps, 3)
            y.append(features[i, 0])               # valor objetivo escalado

        X = np.array(X)
        y = np.array(y)

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        # ============================
        # 4. Train / Test split
        # ============================
        train_size = int(len(X) * train_vol)
        self._train_size = train_size

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        self.scaled_ds = {
            "X": (X_train, X_test),
            "Y": (y_train, y_test)
        }

    def train_data(self, epochs=100, batch_size=360, validation_split=0.01, verbose=1, optimizer=None, loss=None, metrics=None):
        model = self._model

        model.compile(optimizer=optimizer or 'adam', loss=loss or 'mse', metrics=metrics or ['mae'])
        #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            self.scaled_ds['X'][0], self.scaled_ds['Y'][0],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        return history

    def test_model(self):
        X_test = self.scaled_ds['X'][1]
        y_test = self.scaled_ds['Y'][1]
        model = self._model

        scaler = self._scaler
        pred_scaled = model.predict(X_test)

        # Reescalar predicciones y valores reales
        pred_rescaled = scaler.inverse_transform(pred_scaled)
        true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(true_rescaled, pred_rescaled))
        mae = mean_absolute_error(true_rescaled, pred_rescaled)

        print("%RMSE:", rmse * 100)
        print("%MAE:", mae * 100)

        test_dates = self.df[self._date_col][self._n_steps + self._train_size:]

        # Asegurar longitudes iguales
        test_dates = test_dates.iloc[:len(pred_rescaled)]
        self._pred_rescaled = pred_rescaled
        self._true_rescaled = true_rescaled
        self._test_dates = test_dates

    def plot_tests(self):
        plt.figure(figsize=(14,5))
        plt.plot(self.df[self._date_col], self.ds, label="Real", alpha=0.5)
        plt.plot(self._test_dates, self._pred_rescaled, label="Predicción", color="red", linewidth=2)
        plt.title("Predicción LSTM de " + self._target_col)
        plt.xlabel(self._date_col)
        plt.ylabel(self._target_col)
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def save_model(self, file_name:str='model.keras'):
        self._model.save(f'./data/{file_name}.keras')

    def predict_for_date(self, target_date: str):
        """
        Predice el valor para una fecha futura usando predicción autoregresiva.
        """

        target_date = pd.to_datetime(target_date)

        last_date = self.df[self._date_col].iloc[-1]

        if target_date <= last_date:
            raise ValueError("La fecha debe ser futura")

        steps_ahead = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)

        # Última ventana conocida
        values = self.df[self._target_col].values.reshape(-1, 1)

        sin_month = np.sin(2 * np.pi * self.df[self._date_col].dt.month / 12)
        cos_month = np.cos(2 * np.pi * self.df[self._date_col].dt.month / 12)

        features = np.hstack([values, sin_month.values.reshape(-1,1), cos_month.values.reshape(-1,1)])

        # Escalar SOLO la variable objetivo
        scaled_values = self._scaler.transform(values)

        window = features[-self._n_steps:].copy()

        current_date = last_date
        prediction = None

        for _ in range(steps_ahead):
            sin_m, cos_m = self.encode_date(current_date + pd.DateOffset(months=1))

            window_scaled = window.copy()
            window_scaled[:,0] = self._scaler.transform(window[:,0].reshape(-1,1)).flatten()

            X = window_scaled.reshape(1, self._n_steps, 3)

            pred_scaled = self._model.predict(X, verbose=0)
            prediction = self._scaler.inverse_transform(pred_scaled)[0,0]

            # avanzar ventana
            new_row = np.array([prediction, sin_m, cos_m])
            window = np.vstack([window[1:], new_row])

            current_date += pd.DateOffset(months=1)

        return prediction
