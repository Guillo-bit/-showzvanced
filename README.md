# Proyecto de Predicción de Marketing: LTV, CAC y ROMI

## Descripción

Este proyecto tiene como objetivo predecir dos métricas clave en el marketing: **Customer Lifetime Value (LTV)** y **Customer Acquisition Cost (CAC)**. Usando diferentes modelos de Machine Learning, se busca optimizar el rendimiento del marketing calculando el **Return on Marketing Investment (ROMI)** bajo distintos escenarios de reasignación de presupuesto.

Se entrenaron y evaluaron varios modelos, incluyendo **Ridge**, **LightGBM**, y **Linear Regression**, para predecir estas métricas. Además, se generaron gráficos de interpretabilidad usando SHAP y se realizaron simulaciones para evaluar la estrategia de marketing más efectiva.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener los siguientes requisitos instalados:

- **Python 3.x** (preferentemente la versión más reciente).
- **Librerías de Python**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `lightgbm`
  - `matplotlib`
  - `shap`
  - `seaborn`

Para instalar las librerías necesarias, puedes ejecutar:

```bash
pip install -r requirements.txt
```

Estructura de carpetas


```bash
- `data/`
  - `engineered/`
    - `final_dataset.csv`
  - `interim/`
    - `costs.csv`
    - `orders.csv`
    - `visits.csv`
  - `processed/`
    - `features.csv`
  - `raw/`
    - `costs.csv`
    - `orders.csv`
    - `visits.csv`
- `models/`
  - `archivos .pkl`
- `notebooks/`
  - `01_EDA.ipynb`
  - `02_Feature_Engineering.ipynb`
  - `03_Model_Training.ipynb`
  - `Final_Project_Showz_LTV_CAC.ipynb`
- `reports/`
  - `executive_summary.pdf`
  - `figures/`
    - `engineered/`
      - `boxplots.png`
      - `boxplot_LTV_180_marketing.png`
      - `boxplot_LTV_180_num_sessions.png`
      - `comparacion_LTV_180.png`
      - `correlacion.png`
      - `distribuciones.png`
      - `distribucion_LTV_180.png`
      - `distribucion_LTV_imputado.png`
      - `log_LTV_180.png`
      - `usuarios_por_canal_marketing.png`
    - `final/`
      - `Residuals for CAC Model.png`
      - `Residuals for LTV Model.png`
      - `SHAP Value (Impact on Model Output).png`
      - `SHAP Value.png`
    - `raw/`
      - `after_winsorizing_boxplots.png`
      - `after_winsorizing_histograms.png`
      - `before_winsorizing_boxplots.png`
      - `before_winsorizing_histograms.png`
- `requirements.txt`
- `src/`
- `README.ipynb`

```

## Estructura de Proyecto

La estructura de carpetas está organizada para seguir un flujo de trabajo eficiente en el desarrollo de tu proyecto de minería de datos. Aquí se detalla cada carpeta y su propósito:

- **`data/`**: Contiene los datos en diferentes estados:
  - `raw/`: Datos sin procesar que provienen directamente de la fuente.
  - `interim/`: Datos intermedios que se utilizan en pasos de preprocesamiento.
  - `processed/`: Datos que ya han sido limpiados y transformados, listos para el modelado.
  - `engineered/`: Datos con características adicionales o transformadas para alimentar modelos específicos.

- **`models/`**: Aquí se almacenan los modelos entrenados y sus configuraciones. Incluye:
  - Modelos para predecir métricas como CAC (Costo de Adquisición de Clientes) y LTV (Valor de Vida del Cliente).
  - Modelos optimizados para diferentes algoritmos, como Lasso, LightGBM, Ridge, etc.
  - Modelos de stacking que combinan predicciones de múltiples modelos base.

- **`notebooks/`**: Carpeta que contiene los cuadernos Jupyter con el flujo de trabajo:
  - `01_EDA.ipynb`: Análisis exploratorio de datos (EDA).
  - `02_Feature_Engineering.ipynb`: Proceso de ingeniería de características.
  - `03_Model_Training.ipynb`: Entrenamiento de los modelos.
  - `Final_Project_Showz_LTV_CAC.ipynb`: Cuaderno final que integra todo el flujo de trabajo.

- **`reports/`**: Carpeta destinada a almacenar los reportes generados y las visualizaciones.
  - `executive_summary.pdf`: Resumen ejecutivo del análisis.
  - **`figures/`**: Subcarpeta con gráficos generados en diferentes etapas del proyecto:
    - `engineered/`: Gráficos relacionados con los datos transformados.
    - `final/`: Resultados finales del modelo, como residuales y valores SHAP.
    - `raw/`: Gráficos antes y después de la transformación de datos.

- **`requirements.txt`**: Archivo con las dependencias necesarias para ejecutar el proyecto.

- **`src/`**: Contiene el código fuente del proyecto, como funciones y clases reutilizables.

- **`README.ipynb`**: Cuaderno de documentación del proyecto, con detalles de la metodología y el flujo de trabajo.

Cada uno de estos directorios y archivos está diseñado para facilitar el desarrollo organizado, el entrenamiento de modelos y la evaluación, y para asegurar que el análisis y los resultados estén documentados y fácilmente accesibles.

