
"""
02_pipeline_limpieza.py
------------------------------------------------------------
Pipeline de limpieza y preprocesamiento de datos para el
modelo de Machine Learning que predice la potabilidad del agua.

Pasos:
1. Cargar datos originales (water_potability.csv).
2. Imputar valores nulos en ph, Sulfate y Trihalomethanes
   usando KNNImputer (basado en las demás variables numéricas).
3. Separar en train/test de forma estratificada.
4. Escalar las variables numéricas (StandardScaler).
5. Guardar el dataset limpio completo + los splits ya
   procesados, listos para entrenar el modelo.

NOTA: Las variables ficticias (fecha, localidad, tipo_fuente)
generadas en 01_enriquecer_datos.py NO se incluyen aquí
deliberadamente: este pipeline trabaja solo con las variables
fisicoquímicas reales, que son las que tienen relación real
con la potabilidad del agua.

Input:  water_potability.csv
Output: water_potability_limpio.csv
        X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_SEED = 42

# ------------------------------------------------------------
# 0. Carpeta base: la misma carpeta donde está este script.
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. Cargar datos
# ------------------------------------------------------------
df = pd.read_csv(BASE_DIR / "water_potability.csv")

feature_cols = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
target_col = "Potability"

print("Nulos antes de imputar:")
print(df[feature_cols].isna().sum())

# ------------------------------------------------------------
# 2. Imputación con KNNImputer
# ------------------------------------------------------------
# KNNImputer estima cada valor faltante a partir de los k vecinos
# más cercanos (según las demás variables numéricas), lo cual
# suele ser más preciso que imputar con la media/mediana global,
# porque respeta relaciones entre variables.
#
# Importante: se escalan los datos ANTES de calcular los vecinos
# (las variables tienen escalas muy distintas, ej. Solids ~20000
# vs ph ~7), si no, el cálculo de distancia quedaría dominado por
# las variables de mayor magnitud.

scaler_pre = StandardScaler()
X_scaled_for_impute = scaler_pre.fit_transform(df[feature_cols])

knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
X_imputed_scaled = knn_imputer.fit_transform(X_scaled_for_impute)

# Revertir el escalado para volver a las unidades originales
X_imputed = scaler_pre.inverse_transform(X_imputed_scaled)

df_clean = df.copy()
df_clean[feature_cols] = X_imputed

print("\nNulos después de imputar:")
print(df_clean[feature_cols].isna().sum())

# Guardar dataset limpio completo (sin escalar, valores en unidades reales)
df_clean.to_csv(BASE_DIR / "water_potability_limpio.csv", index=False)

# ------------------------------------------------------------
# 3. Split train/test (estratificado por la clase objetivo)
# ------------------------------------------------------------
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"\nTrain: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
print("Distribución de clases en train:")
print(y_train.value_counts(normalize=True))
print("Distribución de clases en test:")
print(y_test.value_counts(normalize=True))

# ------------------------------------------------------------
# 4. Escalado final (para el modelo) — fit SOLO con train
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=feature_cols, index=X_test.index
)

# ------------------------------------------------------------
# 5. Guardar splits y objetos del pipeline
# ------------------------------------------------------------
X_train_scaled.to_csv(BASE_DIR / "X_train.csv", index=False)
X_test_scaled.to_csv(BASE_DIR / "X_test.csv", index=False)
y_train.to_csv(BASE_DIR / "y_train.csv", index=False)
y_test.to_csv(BASE_DIR / "y_test.csv", index=False)

joblib.dump(scaler, BASE_DIR / "scaler.pkl")
joblib.dump(knn_imputer, BASE_DIR / "knn_imputer.pkl")
joblib.dump(scaler_pre, BASE_DIR / "scaler_pre_imputacion.pkl")

print("\nPipeline completado. Archivos generados:")
print("- water_potability_limpio.csv")
print("- X_train.csv / X_test.csv / y_train.csv / y_test.csv")
print("- scaler.pkl / knn_imputer.pkl / scaler_pre_imputacion.pkl")
