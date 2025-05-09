import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# Cargar el dataset final
data_path = '../data/engineered/final_dataset.csv'
df = pd.read_csv(data_path)

# Eliminar la columna 'lifetime_days'
df = df.drop(columns=['lifetime_days'])

# Convertir columnas de fecha
df['first_session'] = pd.to_datetime(df['first_session'])
df['last_session'] = pd.to_datetime(df['last_session'])

# Definir las variables dependientes
target_ltv = 'LTV_180'
target_cac = 'CAC_source_30'

# Filtrar filas con valores conocidos
df_ltv = df[df[target_ltv].notnull()]
df_cac = df[df[target_cac].notnull()]

# Separar temporalmente según la fecha de first_session
def temporal_split(df, date_column, train_end, valid_end):
    train = df[df[date_column] < train_end]
    valid = df[(df[date_column] >= train_end) & (df[date_column] < valid_end)]
    test = df[df[date_column] >= valid_end]
    return train, valid, test

train_ltv, valid_ltv, test_ltv = temporal_split(df_ltv, 'first_session', '2018-02-01', '2018-03-01')
train_cac, valid_cac, test_cac = temporal_split(df_cac, 'first_session', '2018-02-01', '2018-03-01')

# Verificar si alguno de los conjuntos está vacío
for name, part in [('Train LTV', train_ltv), ('Valid LTV', valid_ltv), ('Test LTV', test_ltv),
                   ('Train CAC', train_cac), ('Valid CAC', valid_cac), ('Test CAC', test_cac)]:
    if part.empty:
        print(f"[ADVERTENCIA] El conjunto {name} está vacío.")

# Función para preparar datos
def prepare_xy(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

X_train_ltv, y_train_ltv = prepare_xy(train_ltv, target_ltv)
X_valid_ltv, y_valid_ltv = prepare_xy(valid_ltv, target_ltv)
X_test_ltv, y_test_ltv = prepare_xy(test_ltv, target_ltv)

X_train_cac, y_train_cac = prepare_xy(train_cac, target_cac)
X_valid_cac, y_valid_cac = prepare_xy(valid_cac, target_cac)
X_test_cac, y_test_cac = prepare_xy(test_cac, target_cac)

# Agregar columna binaria is_ltv_imputed
def add_imputation_flag(X):
    imputed_flag = X.isna().astype(int).max(axis=1)  # 1 si hay algún NaN en la fila
    X = X.copy()
    X['is_ltv_imputed'] = imputed_flag
    return X

X_train_ltv = add_imputation_flag(X_train_ltv)
X_valid_ltv = add_imputation_flag(X_valid_ltv)
X_test_ltv = add_imputation_flag(X_test_ltv)

# Imputar valores numéricos
def safe_impute_and_scale(X_train, X_valid, X_test):
    num_cols = X_train.select_dtypes(include=["number"]).columns

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train[num_cols])
    X_valid_imp = imputer.transform(X_valid[num_cols])
    X_test_imp = imputer.transform(X_test[num_cols])

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_valid_scaled, X_test_scaled

X_train_ltv_scaled, X_valid_ltv_scaled, X_test_ltv_scaled = safe_impute_and_scale(X_train_ltv, X_valid_ltv, X_test_ltv)
X_train_cac_scaled, X_valid_cac_scaled, X_test_cac_scaled = safe_impute_and_scale(X_train_cac, X_valid_cac, X_test_cac)

# Modelos base
def entrenar_y_guardar(nombre, modelo, X, y):
    if len(y) == 0:
        print(f"[INFO] No se entrenó {nombre} (datos insuficientes).")
        return None
    modelo.fit(X, y)
    joblib.dump(modelo, f'../models/{nombre}.pkl')
    return modelo

print("Entrenando modelos para LTV_180...")
ltv_linear = entrenar_y_guardar("ltv_linear", LinearRegression(), X_train_ltv_scaled, y_train_ltv)
ltv_ridge = entrenar_y_guardar("ltv_ridge", Ridge(), X_train_ltv_scaled, y_train_ltv)
ltv_lasso = entrenar_y_guardar("ltv_lasso", Lasso(), X_train_ltv_scaled, y_train_ltv)
ltv_sgd = entrenar_y_guardar("ltv_sgd", SGDRegressor(), X_train_ltv_scaled, y_train_ltv)

print("Entrenando modelos para CAC_source_30...")
cac_linear = entrenar_y_guardar("cac_linear", LinearRegression(), X_train_cac_scaled, y_train_cac)
cac_ridge = entrenar_y_guardar("cac_ridge", Ridge(), X_train_cac_scaled, y_train_cac)
cac_lasso = entrenar_y_guardar("cac_lasso", Lasso(), X_train_cac_scaled, y_train_cac)
cac_sgd = entrenar_y_guardar("cac_sgd", SGDRegressor(), X_train_cac_scaled, y_train_cac)

print("Entrenando modelos avanzados...")
ltv_lightgbm = entrenar_y_guardar("ltv_lightgbm", lgb.LGBMRegressor(), X_train_ltv_scaled, y_train_ltv)
cac_lightgbm = entrenar_y_guardar("cac_lightgbm", lgb.LGBMRegressor(), X_train_cac_scaled, y_train_cac)

print("Entrenando ensambladores...")
def entrenar_stacking(nombre, estimators, X, y):
    if len(y) == 0:
        print(f"[INFO] No se entrenó stacking {nombre} (datos insuficientes).")
        return None
    stacking = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
    stacking.fit(X, y)
    joblib.dump(stacking, f'../models/stacking_{nombre}.pkl')
    return stacking

base_ltv = [('linear', ltv_linear), ('ridge', ltv_ridge), ('lasso', ltv_lasso), ('sgd', ltv_sgd), ('lightgbm', ltv_lightgbm)]
base_cac = [('linear', cac_linear), ('ridge', cac_ridge), ('lasso', cac_lasso), ('sgd', cac_sgd), ('lightgbm', cac_lightgbm)]

base_ltv = [(name, model) for name, model in base_ltv if model]
base_cac = [(name, model) for name, model in base_cac if model]

stacking_ltv = entrenar_stacking("ltv", base_ltv, X_train_ltv_scaled, y_train_ltv)
stacking_cac = entrenar_stacking("cac", base_cac, X_train_cac_scaled, y_train_cac)

# Función para calcular métricas
def calcular_metricas(nombre, modelo, X_train, y_train, X_valid, y_valid, X_test, y_test):
    if modelo:
        y_train_pred = modelo.predict(X_train)
        y_valid_pred = modelo.predict(X_valid)
        y_test_pred = modelo.predict(X_test)

        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

        mae_valid = mean_absolute_error(y_valid, y_valid_pred)
        rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        mape_valid = mean_absolute_percentage_error(y_valid, y_valid_pred)

        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

        print(f"\nResultados del modelo {nombre}:")
        print(f"Train - MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, MAPE: {mape_train:.4f}")
        print(f"Validation - MAE: {mae_valid:.4f}, RMSE: {rmse_valid:.4f}, MAPE: {mape_valid:.4f}")
        print(f"Test - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, MAPE: {mape_test:.4f}")

calcular_metricas("LTV_180", stacking_ltv, X_train_ltv_scaled, y_train_ltv, X_valid_ltv_scaled, y_valid_ltv, X_test_ltv_scaled, y_test_ltv)
calcular_metricas("CAC_source_30", stacking_cac, X_train_cac_scaled, y_train_cac, X_valid_cac_scaled, y_valid_cac, X_test_cac_scaled, y_test_cac)

# ========================
# VARIABLES EXPORTABLES
# ========================

__all__ = [
    # Features y targets LTV
    "X_train_ltv_scaled", "X_valid_ltv_scaled", "X_test_ltv_scaled",
    "y_train_ltv", "y_valid_ltv", "y_test_ltv",
    # Features y targets CAC
    "X_train_cac_scaled", "X_valid_cac_scaled", "X_test_cac_scaled",
    "y_train_cac", "y_valid_cac", "y_test_cac",
    # Modelos LTV
    "ltv_linear", "ltv_ridge", "ltv_lasso", "ltv_sgd", "ltv_lightgbm", "stacking_ltv",
    # Modelos CAC
    "cac_linear", "cac_ridge", "cac_lasso", "cac_sgd", "cac_lightgbm", "stacking_cac",
    # Función de métricas
    "calcular_metricas"
]

print("\nEntrenamiento y evaluación completos.")
