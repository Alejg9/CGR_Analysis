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

# ================================================================
# 1. CARGA DE DATOS
# ================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data = pd.read_excel("CGR.xlsx")

print(data.head())
print(data.info())

# Convertir fecha
print(data['Fecha'])
data['Fecha'] = data['Fecha'].astype(str) + '01'
print(data['Fecha'])
data["Fecha"] = pd.to_datetime(data["Fecha"], format='%Y%m%d')

# ================================================================
# 2. SELECCIÓN DE TARGET
# ================================================================
target_col = "tmax(degC)"
dataset = data[[target_col]].values    # shape (N, 1)

# ================================================================
# 3. ESCALADO
# ================================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(dataset)

# ================================================================
# 4. CREAR VENTANAS DESLIZANTES
# ================================================================
n_steps = 12
X, y = [], []

for i in range(n_steps, len(scaled)):
    X.append(scaled[i-n_steps:i, 0])     # ventanas de 60 valores
    y.append(scaled[i, 0])               # siguiente valor objetivo

X = np.array(X).reshape(-1, n_steps, 1)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ================================================================
# 5. TRAIN / TEST SPLIT
# ================================================================
train_size = int(len(X) * 0.9)     # 80% train, 20% test

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train samples:", len(X_train))
print("Test samples :", len(X_test))

# ================================================================
# 6. MODELO LSTM
# ================================================================

model = keras.models.Sequential([
    keras.layers.LSTM(40, return_sequences=True, input_shape=(n_steps, 1), use_cudnn=False, recurrent_dropout=0.2),
    keras.layers.LSTM(80, return_sequences=False, input_shape=(n_steps, 1), use_cudnn=False , recurrent_dropout=0.0),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=360,
    validation_split=0.01,
    verbose=1
)

# ================================================================
# 7. PREDICCIONES DEL MODELO
# ================================================================
pred_scaled = model.predict(X_test)
model.save('model.keras')
# Reescalar predicciones y valores reales
pred_rescaled = scaler.inverse_transform(pred_scaled)
true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# ================================================================
# 8. MÉTRICAS
# ================================================================
rmse = np.sqrt(mean_squared_error(true_rescaled, pred_rescaled))
mae = mean_absolute_error(true_rescaled, pred_rescaled)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}\n")

# ================================================================
# 9. PREPARAR FECHAS PARA EL GRÁFICO
# ================================================================
# Las fechas del test empiezan después de los primeros n_steps + train_size
test_dates = data["Fecha"][n_steps + train_size:]

# Asegurar longitudes iguales
test_dates = test_dates.iloc[:len(pred_rescaled)]

# ================================================================
# 10. GRÁFICO
# ================================================================
plt.figure(figsize=(14,5))
plt.plot(data["Fecha"], dataset, label="Real", alpha=0.5)
plt.plot(test_dates, pred_rescaled, label="Predicción", color="red", linewidth=2)
plt.title("Predicción LSTM de " + target_col)
plt.xlabel("Fecha")
plt.ylabel(target_col)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("prediccion.png", dpi=300, bbox_inches='tight')