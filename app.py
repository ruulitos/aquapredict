

"""
app.py — AquaPredict Dashboard
====================================================================
Sistema de predicción de potabilidad del agua - ML + Streamlit
Proyecto IDS/IS 2026 · ODS 6 (Agua limpia y saneamiento)

Espera los siguientes archivos en la misma carpeta:
- water_potability.csv                (dataset original)
- water_potability_enriquecido.csv    (dataset + variables ficticias)
- water_potability_limpio.csv         (dataset limpio)
- X_train.csv, X_test.csv, y_train.csv, y_test.csv (splits)
- scaler.pkl                          (StandardScaler entrenado)
- knn_imputer.pkl                     (KNNImputer entrenado)
- scaler_pre_imputacion.pkl           (StandardScaler pre-imputation)
- modelo_final.pkl                    (Modelo entrenado)
  O si modelo_final.pkl no existe:
  - modelo_logistic_regression.pkl
  - modelo_random_forest.pkl

Cómo correr:
  streamlit run app.py

El app se adapta automáticamente según qué archivos encuentre.
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================
st.set_page_config(
    page_title="AquaPredict | Potabilidad del Agua",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Las 9 variables fisicoquímicas reales — únicas features del modelo.
FEATURE_COLS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

# Rangos de referencia
RANGOS_REF = {
    "ph":              {"min": 0.0,    "max": 14.0,    "default": 7.0,    "ok": (6.5, 8.5)},
    "Hardness":        {"min": 0.0,    "max": 500.0,   "default": 196.0,  "ok": (0, 300)},
    "Solids":          {"min": 0.0,    "max": 60000.0, "default": 20000.0,"ok": (0, 500)},
    "Chloramines":     {"min": 0.0,    "max": 15.0,    "default": 7.0,    "ok": (0, 4)},
    "Sulfate":         {"min": 0.0,    "max": 500.0,   "default": 333.0,  "ok": (0, 250)},
    "Conductivity":    {"min": 0.0,    "max": 800.0,   "default": 420.0,  "ok": (0, 400)},
    "Organic_carbon":  {"min": 0.0,    "max": 30.0,    "default": 14.0,   "ok": (0, 10)},
    "Trihalomethanes": {"min": 0.0,    "max": 130.0,   "default": 66.0,   "ok": (0, 80)},
    "Turbidity":       {"min": 0.0,    "max": 7.0,     "default": 4.0,    "ok": (0, 5)},
}

UNIDADES = {
    "ph": "", "Hardness": "mg/L", "Solids": "ppm", "Chloramines": "mg/L",
    "Sulfate": "mg/L", "Conductivity": "μS/cm", "Organic_carbon": "ppm",
    "Trihalomethanes": "μg/L", "Turbidity": "NTU",
}


# ==============================================================
# CARGA DE ARTEFACTOS (con fallback inteligente)
# ==============================================================
@st.cache_resource
def cargar_artefactos():
    """Carga artefactos con fallback: primero busca modelo_final.pkl,
    si no existe, busca modelo_random_forest.pkl (el más grande/mejor)."""
    artefactos = {
        "modelo": None, "scaler": None, "knn_imputer": None,
        "scaler_pre": None, "nombre_modelo": "No disponible",
    }
    
    # Intentar cargar modelo (con fallback)
    rutas_modelo = [
        BASE_DIR / "modelo_final.pkl",
        BASE_DIR / "modelo_random_forest.pkl",
        BASE_DIR / "modelo_logistic_regression.pkl",
    ]
    
    for ruta in rutas_modelo:
        if ruta.exists():
            try:
                artefactos["modelo"] = joblib.load(ruta)
                artefactos["nombre_modelo"] = ruta.stem
                break
            except Exception:
                pass
    
    # Cargar otros artefactos
    try:
        if (BASE_DIR / "scaler.pkl").exists():
            artefactos["scaler"] = joblib.load(BASE_DIR / "scaler.pkl")
    except Exception:
        pass
    
    try:
        if (BASE_DIR / "knn_imputer.pkl").exists():
            artefactos["knn_imputer"] = joblib.load(BASE_DIR / "knn_imputer.pkl")
    except Exception:
        pass
    
    try:
        if (BASE_DIR / "scaler_pre_imputacion.pkl").exists():
            artefactos["scaler_pre"] = joblib.load(BASE_DIR / "scaler_pre_imputacion.pkl")
    except Exception:
        pass
    
    return artefactos


@st.cache_data
def cargar_datos():
    """Carga CSVs disponibles con graceful degradation."""
    datos = {"original": None, "enriquecido": None, "limpio": None}

    try:
        if (BASE_DIR / "water_potability.csv").exists():
            datos["original"] = pd.read_csv(BASE_DIR / "water_potability.csv")
    except Exception:
        pass

    try:
        if (BASE_DIR / "water_potability_enriquecido.csv").exists():
            df_enr = pd.read_csv(BASE_DIR / "water_potability_enriquecido.csv")
            if "fecha_muestra" in df_enr.columns:
                try:
                    df_enr["fecha_muestra"] = pd.to_datetime(df_enr["fecha_muestra"])
                except Exception:
                    pass
            datos["enriquecido"] = df_enr
    except Exception:
        pass

    try:
        if (BASE_DIR / "water_potability_limpio.csv").exists():
            datos["limpio"] = pd.read_csv(BASE_DIR / "water_potability_limpio.csv")
    except Exception:
        pass

    return datos


artefactos = cargar_artefactos()
datos = cargar_datos()

modelo_listo = artefactos["modelo"] is not None
pipeline_listo = all(
    artefactos[k] is not None for k in ("scaler", "knn_imputer", "scaler_pre")
)


# ==============================================================
# FUNCIÓN DE PREDICCIÓN
# ==============================================================
def predecir_potabilidad(valores: dict):
    """Predice potabilidad usando el modelo cargado."""
    if not modelo_listo:
        return None, None
    
    X = pd.DataFrame([valores], columns=FEATURE_COLS)
    X_scaled = artefactos["scaler"].transform(X)
    
    try:
        pred = artefactos["modelo"].predict(X_scaled)[0]
        proba = artefactos["modelo"].predict_proba(X_scaled)[0]
        return int(pred), float(proba[1])
    except Exception:
        return None, None


# ==============================================================
# SIDEBAR — NAVEGACIÓN Y ESTADO
# ==============================================================
st.sidebar.title("💧 AquaPredict")
st.sidebar.caption("Proyecto IDS/IS 2026 · ODS 6")

pagina = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔮 Predicción", "📊 Dashboard", "📈 EDA"],
)

st.sidebar.divider()
st.sidebar.subheader("Estado del pipeline")

# Estado visual
estado_modelo = "✅" if modelo_listo else "❌"
estado_scaler = "✅" if artefactos["scaler"] is not None else "❌"
estado_imputer = "✅" if artefactos["knn_imputer"] is not None else "❌"
estado_datos = "✅" if datos["enriquecido"] is not None else "⚠️"

st.sidebar.write(f"Modelo: {estado_modelo} {artefactos['nombre_modelo']}")
st.sidebar.write(f"Scaler: {estado_scaler}")
st.sidebar.write(f"Imputer: {estado_imputer}")
st.sidebar.write(f"Datos enriquecidos: {estado_datos}")

if not modelo_listo:
    st.sidebar.error(
        "⚠️ Modelo no disponible. "
        "Ejecuta `03_entrenar_modelos.py` o asegúrate de que "
        "`modelo_final.pkl` esté en la carpeta."
    )


# ==============================================================
# PÁGINA: INICIO
# ==============================================================
if pagina == "🏠 Inicio":
    st.title("💧 AquaPredict")
    st.subheader("Predicción de potabilidad del agua con Machine Learning")

    st.markdown(
        """
        Sistema de apoyo a decisiones para determinar si el agua
        es apta para consumo humano, basado en parámetros fisicoquímicos.
        
        **ODS 6:** Agua limpia y saneamiento
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    
    n_total = len(datos["original"]) if datos["original"] is not None else 0
    n_potable = (
        int(datos["original"]["Potability"].sum())
        if datos["original"] is not None else 0
    )
    
    col1.metric("Muestras en dataset", f"{n_total:,}")
    col2.metric("Muestras potables", f"{n_potable:,}")
    col3.metric(
        "% Potabilidad",
        f"{(n_potable / n_total * 100):.1f}%" if n_total else "—",
    )
    col4.metric("Variables", len(FEATURE_COLS))

    st.divider()
    st.markdown(
        """
        **Pipeline:**
        1. Enriquecimiento de datos (variables contextuales)
        2. Limpieza: imputación KNN + escalado StandardScaler
        3. Split train/test estratificado
        4. Entrenamiento: Logistic Regression & Random Forest
        5. Dashboard interactivo en Streamlit
        
        **Variables ficticias ≠ Features del modelo:**
        El enriquecimiento agrega fecha, localidad, región y tipo de fuente
        solo para visualización. El modelo **SOLO** usa los 9 parámetros
        fisicoquímicos reales.
        """
    )


# ==============================================================
# PÁGINA: PREDICCIÓN
# ==============================================================
elif pagina == "🔮 Predicción":
    st.title("🔮 Predicción de potabilidad")
    st.caption("Ingresa los parámetros medidos en la muestra.")

    if not modelo_listo or artefactos["scaler"] is None:
        st.error(
            "❌ No se puede predecir. Faltan archivos del pipeline. "
            "Verifica el sidebar para más detalles."
        )
    else:
        with st.form("form_prediccion"):
            st.markdown("##### Parámetros fisicoquímicos")
            cols = st.columns(3)
            valores = {}
            
            for i, feat in enumerate(FEATURE_COLS):
                ref = RANGOS_REF[feat]
                unidad = f" ({UNIDADES[feat]})" if UNIDADES[feat] else ""
                with cols[i % 3]:
                    valores[feat] = st.number_input(
                        f"{feat}{unidad}",
                        min_value=float(ref["min"]),
                        max_value=float(ref["max"]),
                        value=float(ref["default"]),
                        step=(ref["max"] - ref["min"]) / 100,
                    )
            
            enviado = st.form_submit_button("Predecir", type="primary")

        if enviado:
            clase, prob_potable = predecir_potabilidad(valores)

            if clase is not None:
                st.divider()
                col1, col2 = st.columns([1, 2])

                with col1:
                    if clase == 1:
                        st.success("✅ POTABLE")
                    else:
                        st.error("⛔ NO POTABLE")
                    st.metric("Probabilidad potable", f"{prob_potable * 100:.1f}%")
                    st.progress(min(max(prob_potable, 0.0), 1.0))

                with col2:
                    st.markdown("**Parámetros vs. rango deseable:**")
                    filas = []
                    for feat in FEATURE_COLS:
                        ok_min, ok_max = RANGOS_REF[feat]["ok"]
                        val = valores[feat]
                        dentro = ok_min <= val <= ok_max
                        filas.append({
                            "Parámetro": feat,
                            "Valor": round(val, 2),
                            "Rango OK": f"{ok_min}–{ok_max}",
                            "Estado": "✅" if dentro else "⚠️",
                        })
                    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
            else:
                st.error("Error al predecir. Intenta de nuevo.")


# ==============================================================
# PÁGINA: DASHBOARD
# ==============================================================
elif pagina == "📊 Dashboard":
    st.title("📊 Dashboard por comunidad")
    st.caption("Datos enriquecidos con contexto territorial (fecha, región, localidad).")

    df_enr = datos["enriquecido"]

    if df_enr is None:
        st.warning(
            "No se encontró `water_potability_enriquecido.csv`. "
            "Ejecuta `01_enriquecer_datos.py` primero."
        )
    else:
        f1, f2, f3 = st.columns(3)
        
        regiones = sorted(df_enr["region"].dropna().unique()) if "region" in df_enr else []
        fuentes = sorted(df_enr["tipo_fuente"].dropna().unique()) if "tipo_fuente" in df_enr else []

        region_sel = f1.multiselect("Región", regiones, default=regiones)
        fuente_sel = f2.multiselect("Tipo de fuente", fuentes, default=fuentes)
        
        rango_fecha = None
        if "fecha_muestra" in df_enr.columns:
            fecha_min, fecha_max = df_enr["fecha_muestra"].min(), df_enr["fecha_muestra"].max()
            rango_fecha = f3.date_input(
                "Rango de fechas",
                value=(fecha_min, fecha_max),
                min_value=fecha_min,
                max_value=fecha_max,
            )

        df_filtrado = df_enr.copy()
        if region_sel:
            df_filtrado = df_filtrado[df_filtrado["region"].isin(region_sel)]
        if fuente_sel:
            df_filtrado = df_filtrado[df_filtrado["tipo_fuente"].isin(fuente_sel)]
        if rango_fecha and len(rango_fecha) == 2:
            ini, fin = pd.to_datetime(rango_fecha[0]), pd.to_datetime(rango_fecha[1])
            df_filtrado = df_filtrado[
                (df_filtrado["fecha_muestra"] >= ini) & (df_filtrado["fecha_muestra"] <= fin)
            ]

        st.divider()
        m1, m2, m3 = st.columns(3)
        n_f = len(df_filtrado)
        n_pot_f = int(df_filtrado["Potability"].sum()) if n_f else 0
        m1.metric("Muestras", f"{n_f:,}")
        m2.metric("Potables", f"{n_pot_f:,}")
        m3.metric("% Potabilidad", f"{(n_pot_f / n_f * 100):.1f}%" if n_f else "—")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Potabilidad por tipo de fuente**")
            if "tipo_fuente" in df_filtrado.columns and n_f:
                tabla = (
                    df_filtrado.groupby("tipo_fuente")["Potability"]
                    .mean().sort_values(ascending=False) * 100
                )
                st.bar_chart(tabla)

        with c2:
            st.markdown("**Potabilidad por región**")
            if "region" in df_filtrado.columns and n_f:
                tabla = (
                    df_filtrado.groupby("region")["Potability"]
                    .mean().sort_values(ascending=False) * 100
                )
                st.bar_chart(tabla)


# ==============================================================
# PÁGINA: EDA
# ==============================================================
elif pagina == "📈 EDA":
    st.title("📈 Exploración de datos")

    fuente_eda = st.radio(
        "Fuente",
        ["Limpio", "Original"],
        horizontal=True,
    )
    df_eda = datos["limpio"] if fuente_eda == "Limpio" else datos["original"]

    if df_eda is None:
        st.warning("Datos no disponibles.")
    else:
        st.write(f"**{df_eda.shape[0]:,}** filas × **{df_eda.shape[1]}** columnas")

        tab1, tab2, tab3 = st.tabs(["Resumen", "Distribución", "Nulos"])

        with tab1:
            st.dataframe(df_eda.describe().T, use_container_width=True)

        with tab2:
            var_sel = st.selectbox("Variable", FEATURE_COLS)
            colA, colB = st.columns(2)
            
            with colA:
                st.markdown(f"**Histograma — {var_sel}**")
                st.bar_chart(
                    df_eda[var_sel].dropna().value_counts(bins=20).sort_index()
                )
            
            with colB:
                if "Potability" in df_eda.columns:
                    st.markdown(f"**{var_sel} por potabilidad**")
                    st.bar_chart(df_eda.groupby("Potability")[var_sel].mean())

        with tab3:
            nulos = df_eda.isna().sum()
            nulos = nulos[nulos > 0]
            if nulos.empty:
                st.success("Sin valores nulos.")
            else:
                st.bar_chart(nulos)


st.sidebar.divider()
st.sidebar.caption("IDS/IS 2026 · UCN · 2026")

