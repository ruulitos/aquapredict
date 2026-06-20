"""
03_entrenar_modelos.py
------------------------------------------------------------
Entrena y compara dos modelos de clasificación para predecir
la potabilidad del agua, usando los datos ya limpios y escalados
generados por 02_pipeline_limpieza.py.

Modelos entrenados (ambos se guardan para usarlos en el
dashboard de Streamlit con un selector, y así poder comparar
y decidir en cuál confiar):
- Logistic Regression: mejor F1-score / mejor recall para
  detectar agua potable (menos casos potables pasan
  desapercibidos), aunque su accuracy global es más bajo.
- Random Forest: mejor accuracy/precision (más conservador
  al predecir "potable", pero deja pasar más falsos negativos).

Para cada modelo se reportan: Accuracy, Precision, Recall, F1-score
y la matriz de confusión.

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: modelo_logistic_regression.pkl
        modelo_random_forest.pkl
        resultados_modelos.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. Cargar datos ya procesados
# ------------------------------------------------------------
X_train = pd.read_csv(BASE_DIR / "X_train.csv")
X_test = pd.read_csv(BASE_DIR / "X_test.csv")
y_train = pd.read_csv(BASE_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(BASE_DIR / "y_test.csv").squeeze()

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ------------------------------------------------------------
# 2. Definir modelos a comparar
# ------------------------------------------------------------
modelos = {
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_SEED,
        class_weight="balanced", n_jobs=-1
    ),
}

# ------------------------------------------------------------
# 3. Entrenar, evaluar y comparar
# ------------------------------------------------------------
resultados = []
modelos_entrenados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    resultados.append({
        "Modelo": nombre,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
    })

    modelos_entrenados[nombre] = modelo

    print(f"\n{'='*55}")
    print(f"Modelo: {nombre}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["No potable", "Potable"]))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 4. Tabla comparativa
# ------------------------------------------------------------
df_resultados = pd.DataFrame(resultados).sort_values("F1-score", ascending=False)
df_resultados.to_csv(BASE_DIR / "resultados_modelos.csv", index=False)

print(f"\n{'='*55}")
print("COMPARACIÓN FINAL DE MODELOS")
print(f"{'='*55}")
print(df_resultados.to_string(index=False))

# ------------------------------------------------------------
# 5. Guardar los DOS modelos que se mostrarán en el dashboard
#    de Streamlit (selector lateral para comparar ambos):
#    - Logistic Regression: mejor F1-score (mejor balance/recall
#      para detectar agua potable, aunque accuracy más bajo)
#    - Random Forest: mejor accuracy/precision (más conservador,
#      pero pierde más casos potables reales)
# ------------------------------------------------------------
joblib.dump(modelos_entrenados["Logistic Regression"], BASE_DIR / "modelo_logistic_regression.pkl")
joblib.dump(modelos_entrenados["Random Forest"], BASE_DIR / "modelo_random_forest.pkl")

print("\nModelos guardados para el dashboard de Streamlit:")
print("- modelo_logistic_regression.pkl  (mejor F1-score / mejor recall clase Potable)")
print("- modelo_random_forest.pkl        (mejor accuracy / precision)")
print("\nAdemás necesitarás en el repo de GitHub:")
print("- scaler.pkl (para escalar nuevos datos de entrada)")
print("- knn_imputer.pkl + scaler_pre_imputacion.pkl (si hay valores faltantes)")
print("\nEn Streamlit, un selector (st.sidebar.radio o selectbox) permitirá")
print("elegir qué modelo usar para la predicción y mostrar sus métricas")
print("comparativas, dejando al usuario decidir en cuál confiar.")
