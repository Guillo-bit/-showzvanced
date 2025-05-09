import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import warnings
import math
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from utils import simulate_romi_from_real_values

warnings.filterwarnings("ignore")

# ======================
# Cargar datos
# ======================
df = pd.read_csv('../data/engineered/final_dataset.csv')

if 'lifetime_days' in df.columns:
    df = df.drop(columns=['lifetime_days'])

df['first_session'] = pd.to_datetime(df['first_session'])
df['last_session'] = pd.to_datetime(df['last_session'])

# ======================
# Filtrar targets
# ======================
df_ltv = df[df['LTV_180'].notnull()]
df_cac = df[df['CAC_source_30'].notnull()]

# ======================
# Train split (manual)
# ======================
train_ltv = df_ltv[df_ltv['first_session'] < '2018-02-01']
train_cac = df_cac[df_cac['first_session'] < '2018-02-01']

def prepare_xy(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    for col in X.select_dtypes(include='datetime64').columns:
        X[col] = X[col].astype(np.int64) // 10**9
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y

X_ltv, y_ltv = prepare_xy(train_ltv, 'LTV_180')
X_cac, y_cac = prepare_xy(train_cac, 'CAC_source_30')

# ======================
# Definir modelos
# ======================
def modelos_base():
    return {
        "linear": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
        "sgd": make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000)),
        "lightgbm": lgb.LGBMRegressor(n_estimators=100)
    }

# ======================
# Evaluar e imprimir resultados
# ======================
def plot_feature_importance(model, X_train, top_n=5):
    """
    Mostrar la importancia de las características usando SHAP
    """
    # Extraer modelo final si es pipeline
    if isinstance(model, Pipeline):
        final_model = model[-1]
    else:
        final_model = model

    # Elegir el explicador adecuado
    if isinstance(final_model, lgb.LGBMRegressor):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_train, check_additivity=False)
    else:
        explainer = shap.Explainer(final_model, X_train)
        shap_values = explainer(X_train)  # sin check_additivity

    shap.summary_plot(shap_values, X_train, max_display=top_n)
    plt.show()


def analyze_residuals(y_true, y_pred, model_name):
    """
    Analizar los errores sistemáticos a través de los residuos
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals for {model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

def simulate_marketing_budget(investment_scenario, ltv_predictions, cac_predictions, source_assignment):
    """
    Simula cambios de presupuesto con un +10% en una fuente frente a redistribución proporcional.
    
    investment_scenario: array de presupuesto base por fuente (ej. [0.3, 0.3, 0.2, 0.2])
    ltv_predictions: predicciones de LTV para cada registro (forma: [n_samples])
    cac_predictions: predicciones de CAC para cada registro (forma: [n_samples])
    source_assignment: vector de enteros [0,1,2,3,...] de forma [n_samples] indicando la fuente por registro
    """
    # Escenario A: Aumentamos 10% en fuente 0
    scenario_a = investment_scenario.copy()
    scenario_a[0] *= 1.10

    # Escenario B: Redistribución proporcional del 10% adicional
    total = np.sum(investment_scenario) + 0.1 * investment_scenario[0]
    scenario_b = investment_scenario * (1 + 0.1 * (investment_scenario[0] / np.sum(investment_scenario))) / total

    # Cálculo de ROMI para ambos escenarios
    romi_a = 0
    romi_b = 0
    for i in range(len(investment_scenario)):
        mask = source_assignment == i
        if np.sum(cac_predictions[mask]) > 0:
            romi_a += scenario_a[i] * np.sum(ltv_predictions[mask]) / np.sum(cac_predictions[mask])
            romi_b += scenario_b[i] * np.sum(ltv_predictions[mask]) / np.sum(cac_predictions[mask])

    # Mostrar resultados
    print(f"ROMI para escenario A (10% incremento en Fuente A): {romi_a:.4f}")
    print(f"ROMI para escenario B (Redistribución proporcional): {romi_b:.4f}")
    print(f"Mejor estrategia recomendada: {'Aumento de presupuesto en Fuente A' if romi_a > romi_b else 'Redistribución proporcional'}")


# ======================
# Entrenar y elegir mejor modelo
# ======================
def entrenar_y_elegir(X, y, modelos_dict, nombre):
    os.makedirs('../models', exist_ok=True)
    resultados = []

    for nombre_modelo, modelo in modelos_dict.items():
        modelo.fit(X, y)
        pred = modelo.predict(X)
        rmse = math.sqrt(mean_squared_error(y, pred))
        resultados.append((nombre_modelo, modelo, rmse))
        print(f"{nombre} - {nombre_modelo}: RMSE = {rmse:.4f}")

    mejor = sorted(resultados, key=lambda x: x[2])[0]
    joblib.dump(mejor[1], f'../models/{nombre}_best_model.pkl')
    joblib.dump(X.columns.tolist(), f'../models/{nombre}_columns.pkl')
    print(f"\nModelo guardado para {nombre}: {mejor[0]} (RMSE: {mejor[2]:.4f})")
    return mejor

# ======================
# Ejecutar
# ======================
print("\nEvaluando LTV_180...")
mejor_ltv = entrenar_y_elegir(X_ltv, y_ltv, modelos_base(), 'ltv')

# Análisis de la importancia de las características
plot_feature_importance(mejor_ltv[1], X_ltv)

# Análisis de residuos
ltv_predictions = mejor_ltv[1].predict(X_ltv)
analyze_residuals(y_ltv, ltv_predictions, "LTV Model")

print("\nEvaluando CAC_source_30...")
mejor_cac = entrenar_y_elegir(X_cac, y_cac, modelos_base(), 'cac')

# Análisis de la importancia de las características
plot_feature_importance(mejor_cac[1], X_cac)

# Análisis de residuos
cac_predictions = mejor_cac[1].predict(X_cac)
analyze_residuals(y_cac, cac_predictions, "CAC Model")

print("\nEjecutando simulación de ROMI...")
simulate_marketing_budget(np.array([0.3, 0.3, 0.2, 0.2]), ltv_predictions, cac_predictions)

print("\nSimulación de ROMI completada.")

# Simulación de ROMI desde valores reales
df = pd.read_csv('../data/engineered/final_dataset.csv')
simulate_romi_from_real_values(df)
print("Simulación de ROMI completada.")
