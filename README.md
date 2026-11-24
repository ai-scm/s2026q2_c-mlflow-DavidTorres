# Taller Hands-On: MLflow + Docker – Del dato al modelo desplegado

**Tutorial base:** [mtpatter/mlflow-tutorial](https://github.com/mtpatter/mlflow-tutorial)

---

## 1. Objetivo del Taller
Implementar de forma práctica y reproducible el ciclo de vida completo de un modelo de Machine Learning (clasificación binaria con `scikit-learn` sobre el dataset *Breast Cancer*), abarcando desde la inspección inicial de datos hasta su exposición como una API REST versionada mediante **MLflow** y **Docker**.

---

## 🔄 2. Conexión: Data Lifecycle vs. ML Lifecycle

Al pasar de flujos tradicionales de ingeniería de datos a MLOps, las etapas fundamentales evolucionan de la siguiente manera:

| Fase en Data Lifecycle | Fase en ML Lifecycle | ¿Qué se mantiene, cambia o amplía? | Implementación en el Taller |
| :--- | :--- | :--- | :--- |
| **Ingestión** | Extracción y partición (`train`/`test`) | **Se mantiene** la lectura de fuentes estructuradas, pero se exige separar los datos de prueba para evaluar sin fuga de datos (*data leakage*). | `sklearn.datasets` (`test.csv`) |
| **Transformación** | Entrenamiento y empaquetado | **Cambia** el artefacto de salida: se serializa un modelo entrenado junto con su entorno de ejecución y metadatos. | `clf-train.py` / `clf-train-registry.py` |
| **Almacenamiento / Catálogo** | Experiment Tracking y Model Registry | **Se amplía** el gobierno y versionado: se registran ejecuciones (*runs*), parámetros, métricas, artefactos y alias de despliegue (`Staging`). | MLflow Server + SQLite (`mlflow.db`) |
| **Consumo (BI / SQL)** | Despliegue (Serving) e Inferencia | **Cambia** el patrón de consumo: pasamos de consultas analíticas batch a predicciones en tiempo real vía peticiones HTTP POST. | `mlflow models serve` (`:1234`) + `curl` |

---

## 3. Estructura del Repositorio

```text
.
├── README.md                              # Documentación técnica y guía paso a paso del taller
├── .gitignore                             # Exclusión de archivos temporales de Python, SQLite y SO
└── mlflow-tutorial/                       # Código fuente y artefactos del ejercicio hands-on
    ├── Dockerfile                         # Definición de imagen reproducible para entrenamiento y serving
    ├── requirements.txt                   # Dependencias Python (incluye ajuste sqlalchemy<2.0.36)
    ├── docker-compose-no-registry.yml     # Orquestación de entrenamiento y API sin registro central
    ├── docker-compose.yml                 # Orquestación completa: MLflow Server + Training + Serving
    ├── clf-train.py                       # Script de entrenamiento y guardado local en disco
    ├── clf-train-registry.py              # Script de entrenamiento, tracking y registro con alias Staging
    ├── predict.sh                         # Cliente HTTP (curl) para consumir el endpoint /invocations
    ├── clf-model/                         # Artefacto del modelo empaquetado localmente
    ├── mlflow.db                          # Base de datos SQLite del backend de MLflow
    ├── mlflow-artifact-root/              # Almacenamiento de artefactos versionados por run
    └── test.csv                           # Conjunto de prueba generado para inferencia
```

---

## 4. Desarrollo Paso a Paso

### Paso 1: Preparación del entorno y clonación
Se clonó el repositorio base dentro del espacio de trabajo y se verificó la disponibilidad de Docker y Docker Compose para ejecutar todo el ciclo de vida de MLflow en contenedores aislados:
```bash
git clone https://github.com/mtpatter/mlflow-tutorial.git
cd mlflow-tutorial
docker --version && docker compose version
```

### Paso 2: Inspección del esquema de datos y lógica de entrenamiento
Se revisó `clf-train.py` para identificar cómo se procesa el dataset *Breast Cancer* (30 *features* numéricas) y cómo se prepara el modelo para inferencia:
* Se entrena un `RandomForestClassifier` de `scikit-learn`.
* Se sobrescribe el método `.predict()` del modelo (`model.predict = overwrite_predict(model.predict_proba)`) para que el endpoint REST de MLflow devuelva directamente las **probabilidades predichas** en lugar de únicamente la clase binaria.

### Paso 3: Entrenamiento y despliegue SIN Model Registry
Se levantó el primer escenario contenedorizado para validar la paridad entre el entorno de entrenamiento y el servidor de inferencia sin utilizar base de datos ni interfaz web:
```bash
docker compose -f docker-compose-no-registry.yml up --build
```
* `trainmodel`: Entrenó el clasificador, generó `test.csv` y serializó el paquete MLflow en la carpeta local `clf-model/` (`MLmodel`, `conda.yaml`, `model.pkl`, `requirements.txt`).
* `servemodel`: Levantó un servidor Gunicorn exponiendo el modelo local en `[http://0.0.0.0:1234](http://0.0.0.0:1234)`.

### Paso 4: Prueba de inferencia sobre la API REST
Con el servicio activo en el puerto `1234`, se enviaron los registros de `test.csv` mediante una petición `POST` al endpoint `/invocations`:
```bash
chmod +x predict.sh
./predict.sh test.csv
```
El servicio respondió con un arreglo JSON (`{"predictions": [0.5074, 0.0, 0.9837, ...]}`) confirmando la inferencia en tiempo real sin depender de librerías instaladas en el sistema anfitrión. Posteriormente se detuvieron los contenedores (`docker compose -f docker-compose-no-registry.yml down`) y se limpiaron los artefactos temporales.

### Paso 5: Ciclo completo con Tracking & Model Registry (y resolución de dependencias)
Se ejecutó el flujo completo con servidor MLflow centralizado (`docker-compose.yml`):

1. **Troubleshooting de dependencias:** Al construir la imagen con Python 3.12, `pip` instaló una versión reciente de `SQLAlchemy` incompatible con el backend SQLite de MLflow 2.x (`ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool' from 'sqlalchemy.pool'`). Se fijó la versión compatible en `requirements.txt` y se reconstruyó el stack:
   ```bash
   echo "sqlalchemy<2.0.36" >> requirements.txt
   docker compose -f docker-compose.yml down
   docker compose -f docker-compose.yml up --build
   ```
2. **Orquestación de los 3 servicios:**
   * `server`: Inicializó el servidor de Tracking y Registry con backend SQLite (`mlflow.db`) y UI web en `http://localhost:8000`.
   * `trainmodel`: Registró la ejecución bajo el experimento `my-experiment`, creó la **versión 1** de `clf-model` en el Model Registry y le asignó el alias **`Staging`**.
   * `servemodel`: Desplegó en el puerto `1234` el modelo resuelto dinámicamente mediante el URI `models:/clf-model@Staging`.

### Paso 6: Verificación en UI de MLflow e inferencia gobernada por alias
* **Interfaz Web (`http://localhost:8000`):** Se validó el registro de parámetros, métricas y artefactos del *run* en `my-experiment`, así como la gobernanza del modelo `clf-model` (v1) bajo el alias `Staging`.
* **Consumo del endpoint:** Se ejecutó nuevamente `./predict.sh test.csv`, comprobando que la API sirve transparentemente la versión asociada al alias `Staging`.
* **Cierre de recursos:** Se detuvieron y eliminaron los contenedores y redes virtuales con `docker compose -f docker-compose.yml down`.

---

## 5. Equivalencia en Arquitectura Cloud (AWS)

El flujo contenedorizado implementado en este taller se traduce directamente a servicios administrados en AWS:
* **MLflow Tracking y Model Registry (`mlflow.db`)** $\rightarrow$ *Amazon SageMaker Model Registry* / *SageMaker Managed MLflow*.
* **Almacenamiento de artefactos (`mlflow-artifact-root/`)** $\rightarrow$ *Amazon S3*.
* **Contenedor Docker de Serving (`localhost:1234`)** $\rightarrow$ *Amazon SageMaker Endpoints* o *AWS ECS / Fargate* desplegando la imagen desde *Amazon ECR*.
