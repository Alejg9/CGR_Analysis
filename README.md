# CGR_Analysis
## Descripción

Análisis exploratorio y comparación de métodos predictivos (ARIMA, SARIMA, Deep Learning con TensorFlow).

El entorno está preparado para ejecutarse con gráficas AMD, para ejecución mediante NVIDIA o CPU leer la documentación.

Para ejecutar el entorno con AMD es necesario hacerlo en un Sistema Operativo Linux, ya que ROCM no tiene soporte para Windows/WSL.
### Versiones
Python==3.10\
Tensorflow==2.20

### Entorno Utilizado

Ubuntu 24.04 LTS\
AMD Radeon RX 9070XT

### Instrucciones de levantamiento del entorno.

1. Para la ejecución del entorno es necesario instalar docker.
2. Dentro de la ruta del repositorio, ejecutamos\
`docker compose up -d`
3. Accedemos a `http://localhost:8888`
4. Dentro accederemos a `/notebooks`, donde podremos ejecutar el código.

## NVIDIA

1. Cambiamos el nombre del archivo `docker-compose.nvidia.yaml` a `docker-compose.yaml`
2. Ejecutamos el comando\
`docker compose up -d`
3. Luego dentro del código seteamos la opción `use_cudnn = True`, esto habilitará el uso de núcleos CUDA.

## CPU