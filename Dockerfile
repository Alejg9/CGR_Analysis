# Base image con TensorFlow + ROCm
FROM rocm/tensorflow:rocm7.1.1-py3.10-tf2.20-dev

# Instalar dependencias adicionales (ej. PyTorch para GPU)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    python3 -m pip install --no-cache-dir \
        notebook \
        jupyterlab \
        polars \
        fastexcel \
        numpy \
        matplotlib \
        scikit-learn \
        seaborn \
        openpyxl \
        plotly \
        statsmodels \
        prophet

# Crear directorio para notebooks (si no existe)
RUN mkdir -p /notebooks /notebooks/libs /notebooks/data

# Configurar Jupyter (opcional: ajustar para autenticación)
ENV JUPYTER_CONFIG_DIR=/root/.jupyter
ENV JUPYTER_PORT=8888