"""
app.py
------------------------------------------------------------
Dashboard de Streamlit — Proyecto IDS/IS 2026 (ODS 6: Agua Limpia
y Saneamiento)

Predicción de potabilidad del agua a partir de parámetros
fisicoquímicos, usando el pipeline construido en:
  01_enriquecer_datos.py   -> water_potability_enriquecido.csv
  02_pipeline_limpieza.py  -> scaler_pre_imputacion.pkl, knn_imputer.pkl,
                               scaler.pkl, water_potability_limpio.csv
  03_entrenar_modelos.py   -> modelo_final.pkl

IMPORTANTE — variables ficticias vs. variables reales:
  Las columnas simuladas en el enriquecimiento (fecha_muestra,
  localidad, region, tipo_fuente, id_muestra) NUNCA se usan como
  input del modelo. Solo sirven para dar contexto narrativo al
  dashboard (filtros, mapas, series de tiempo). El modelo predice
  exclusivamente a partir de las 9 variables fisicoquímicas reales
  del dataset original de Kaggle.

Cómo correrlo:
  streamlit run app.py

Estructura de carpeta esperada (todos los paths son automáticos,
relativos a este archivo, gracias a pathlib):
  app.py
  water_potability.csv
  water_potability_enriquecido.csv   (opcional, para pestaña Dashboard)
  water_potability_limpio.csv        (opcional, para pestaña EDA)
  scaler_pre_imputacion.pkl
  knn_imputer.pkl
  scaler.pkl
  modelo_final.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

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

# Rangos de referencia (OMS / guías generales) usados solo para
# valores por defecto del formulario y para colorear el dashboard.
# No se usan dentro del modelo.
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
# CARGA DE ARTEFACTOS (cacheada para no recargar en cada interacción)
# ==============================================================
@st.cache_resource
def cargar_artefactos():
    """Carga modelo y objetos del pipeline si existen en disco.
    Devuelve un dict con lo que SÍ se pudo cargar, y deja en None
    lo que falte, para que la app avise en vez de romperse."""
    artefactos = {
        "modelo": None, "scaler": None, "knn_imputer": None,
        "scaler_pre": None,
    }
    rutas = {
        "modelo": BASE_DIR / "modelo_final.pkl",
        "scaler": BASE_DIR / "scaler.pkl",
        "knn_imputer": BASE_DIR / "knn_imputer.pkl",
        "scaler_pre": BASE_DIR / "scaler_pre_imputacion.pkl",
    }
    for nombre, ruta in rutas.items():
        if ruta.exists():
            try:
                artefactos[nombre] = joblib.load(ruta)
            except Exception:
                artefactos[nombre] = None
    return artefactos


@st.cache_data
def cargar_datos():
    """Carga los CSV disponibles. Si falta el enriquecido o el
    limpio, cae de vuelta al CSV original para que la app no se rompa."""
    datos = {"original": None, "enriquecido": None, "limpio": None}

    ruta_original = BASE_DIR / "water_potability.csv"
    if ruta_original.exists():
        datos["original"] = pd.read_csv(ruta_original)

    ruta_enriquecido = BASE_DIR / "water_potability_enriquecido.csv"
    if ruta_enriquecido.exists():
        df_enr = pd.read_csv(ruta_enriquecido)
        if "fecha_muestra" in df_enr.columns:
            df_enr["fecha_muestra"] = pd.to_datetime(df_enr["fecha_muestra"])
        datos["enriquecido"] = df_enr

    ruta_limpio = BASE_DIR / "water_potability_limpio.csv"
    if ruta_limpio.exists():
        datos["limpio"] = pd.read_csv(ruta_limpio)

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
    """Recibe un dict {feature: valor} con las 9 variables reales,
    aplica el mismo pipeline de preprocesamiento (imputer + scaler)
    usado en entrenamiento, y devuelve (clase_predicha, probabilidad).

    No se necesita imputar en este flujo porque el usuario completa
    todos los campos del formulario, pero se deja la transformación
    de escalado idéntica a la de entrenamiento para que el modelo
    reciba datos en la misma distribución que vio al entrenar.
    """
    X = pd.DataFrame([valores], columns=FEATURE_COLS)
    X_scaled = artefactos["scaler"].transform(X)
    pred = artefactos["modelo"].predict(X_scaled)[0]
    proba = artefactos["modelo"].predict_proba(X_scaled)[0]
    return int(pred), float(proba[1])  # proba[1] = prob. de ser potable


# ==============================================================
# SIDEBAR — NAVEGACIÓN Y ESTADO DEL PIPELINE
# ==============================================================
st.sidebar.title("💧 AquaPredict")
st.sidebar.caption("Proyecto IDS/IS 2026 · ODS 6 — Agua limpia y saneamiento")

pagina = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔮 Predicción", "📊 Dashboard comunidad", "📈 Exploración de datos (EDA)"],
)

st.sidebar.divider()
st.sidebar.subheader("Estado del pipeline")
st.sidebar.write("Modelo entrenado:", "✅" if modelo_listo else "❌ falta `modelo_final.pkl`")
st.sidebar.write("Scaler:", "✅" if artefactos["scaler"] is not None else "❌ falta `scaler.pkl`")
st.sidebar.write("KNN Imputer:", "✅" if artefactos["knn_imputer"] is not None else "❌ falta `knn_imputer.pkl`")
st.sidebar.write("Dataset enriquecido:", "✅" if datos["enriquecido"] is not None else "❌ falta CSV enriquecido")

if not modelo_listo:
    st.sidebar.warning(
        "Aún no se detecta `modelo_final.pkl`. Corre `03_entrenar_modelos.py` "
        "y colócalo en esta misma carpeta para activar la pestaña de Predicción."
    )


# ==============================================================
# PÁGINA: INICIO
# ==============================================================
if pagina == "🏠 Inicio":
    st.title("💧 AquaPredict")
    st.subheader("Sistema de apoyo a la decisión para potabilidad del agua")

    st.markdown(
        """
        Este dashboard es el prototipo funcional del proyecto, alineado al
        **ODS 6 (Agua limpia y saneamiento)**. Permite estimar, a partir de
        parámetros fisicoquímicos medidos en una muestra de agua, si esta
        es **apta o no apta para consumo humano**, usando un modelo de
        Machine Learning entrenado sobre datos reales de calidad del agua.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    n_total = len(datos["original"]) if datos["original"] is not None else 0
    n_potable = (
        int(datos["original"]["Potability"].sum())
        if datos["original"] is not None else 0
    )
    col1.metric("Muestras en el dataset base", f"{n_total:,}")
    col2.metric("Muestras potables", f"{n_potable:,}")
    col3.metric(
        "% Potabilidad histórica",
        f"{(n_potable / n_total * 100):.1f}%" if n_total else "—",
    )
    col4.metric("Variables fisicoquímicas", len(FEATURE_COLS))

    st.divider()
    st.markdown(
        """
        **¿Cómo se construyó este modelo?**

        1. **Datos** — dataset de calidad del agua (Kaggle), 9 parámetros
           fisicoquímicos + variable objetivo `Potability`.
        2. **Pipeline de limpieza** — imputación de nulos en `ph`, `Sulfate`
           y `Trihalomethanes` con `KNNImputer`, separación train/test
           estratificada, escalado con `StandardScaler`.
        3. **Enriquecimiento contextual** — se simulan variables como fecha,
           localidad y tipo de fuente para alimentar el dashboard. Estas
           variables **no se usan en el modelo**, ya que son sintéticas y
           no tienen relación real con la potabilidad del agua.
        4. **Modelo** — comparación de Regresión Logística, Random Forest y
           Gradient Boosting; se selecciona el de mejor F1-score dado el
           desbalance de clases.

        Usa el menú lateral para predecir una muestra nueva, explorar el
        dashboard simulado por comunidad/localidad, o revisar el análisis
        exploratorio de los datos.
        """
    )

# ==============================================================
# PÁGINA: PREDICCIÓN
# ==============================================================
elif pagina == "🔮 Predicción":
    st.title("🔮 Predicción de potabilidad")
    st.caption("Ingresa los parámetros medidos en la muestra de agua.")

    if not modelo_listo or artefactos["scaler"] is None:
        st.error(
            "No se puede predecir todavía: faltan archivos del pipeline "
            "(`modelo_final.pkl` y/o `scaler.pkl`) en esta carpeta. "
            "Ejecuta primero `02_pipeline_limpieza.py` y `03_entrenar_modelos.py`."
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
                        key=f"input_{feat}",
                    )
            enviado = st.form_submit_button("Predecir potabilidad", type="primary")

        if enviado:
            clase, prob_potable = predecir_potabilidad(valores)

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if clase == 1:
                    st.success("✅ Agua APTA para consumo humano")
                else:
                    st.error("⛔ Agua NO APTA para consumo humano")
                st.metric("Probabilidad de ser potable", f"{prob_potable * 100:.1f}%")
                st.progress(min(max(prob_potable, 0.0), 1.0))

            with res_col2:
                st.markdown("##### Comparación con rangos de referencia")
                filas = []
                for feat in FEATURE_COLS:
                    ok_min, ok_max = RANGOS_REF[feat]["ok"]
                    val = valores[feat]
                    dentro = ok_min <= val <= ok_max
                    filas.append({
                        "Parámetro": feat,
                        "Valor ingresado": round(val, 2),
                        "Rango deseable": f"{ok_min}–{ok_max}",
                        "Estado": "✅ Dentro" if dentro else "⚠️ Fuera",
                    })
                st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

            st.caption(
                "Nota: el resultado es una estimación estadística basada en "
                "datos históricos, no reemplaza un análisis de laboratorio "
                "certificado."
            )

# ==============================================================
# PÁGINA: DASHBOARD COMUNIDAD (usa el CSV enriquecido)
# ==============================================================
elif pagina == "📊 Dashboard comunidad":
    st.title("📊 Dashboard por comunidad")
    st.caption(
        "Visualización con contexto territorial (fecha, localidad, tipo de "
        "fuente). Estas variables son simuladas para fines de presentación "
        "y **no** forman parte de las variables predictoras del modelo."
    )

    df_enr = datos["enriquecido"]

    if df_enr is None:
        st.warning(
            "No se encontró `water_potability_enriquecido.csv` en esta carpeta. "
            "Corre `01_enriquecer_datos.py` para generarlo."
        )
    else:
        # ---- Filtros ----
        f1, f2, f3 = st.columns(3)
        regiones = sorted(df_enr["region"].dropna().unique()) if "region" in df_enr else []
        fuentes = sorted(df_enr["tipo_fuente"].dropna().unique()) if "tipo_fuente" in df_enr else []

        region_sel = f1.multiselect("Región", regiones, default=regiones)
        fuente_sel = f2.multiselect("Tipo de fuente", fuentes, default=fuentes)
        rango_fecha = None
        if "fecha_muestra" in df_enr.columns:
            fecha_min, fecha_max = df_enr["fecha_muestra"].min(), df_enr["fecha_muestra"].max()
            rango_fecha = f3.date_input(
                "Rango de fechas", value=(fecha_min, fecha_max),
                min_value=fecha_min, max_value=fecha_max,
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
        m1.metric("Muestras filtradas", f"{n_f:,}")
        m2.metric("Muestras potables", f"{n_pot_f:,}")
        m3.metric("% Potabilidad", f"{(n_pot_f / n_f * 100):.1f}%" if n_f else "—")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Potabilidad por tipo de fuente")
            if "tipo_fuente" in df_filtrado.columns and n_f:
                tabla = (
                    df_filtrado.groupby("tipo_fuente")["Potability"]
                    .mean().sort_values(ascending=False) * 100
                )
                st.bar_chart(tabla)
            else:
                st.info("Sin datos para mostrar con los filtros actuales.")

        with c2:
            st.markdown("##### Potabilidad por región")
            if "region" in df_filtrado.columns and n_f:
                tabla = (
                    df_filtrado.groupby("region")["Potability"]
                    .mean().sort_values(ascending=False) * 100
                )
                st.bar_chart(tabla)
            else:
                st.info("Sin datos para mostrar con los filtros actuales.")

        if "fecha_muestra" in df_filtrado.columns and n_f:
            st.markdown("##### Tendencia temporal de potabilidad (% mensual)")
            serie = (
                df_filtrado.set_index("fecha_muestra")
                .resample("ME")["Potability"]
                .mean() * 100
            )
            st.line_chart(serie)

        st.markdown("##### Detalle de muestras")
        cols_mostrar = [c for c in [
            "id_muestra", "fecha_muestra", "localidad", "region",
            "tipo_fuente", "ph", "Sulfate", "Trihalomethanes", "Potability",
        ] if c in df_filtrado.columns]
        st.dataframe(df_filtrado[cols_mostrar], hide_index=True, use_container_width=True, height=300)

# ==============================================================
# PÁGINA: EDA
# ==============================================================
elif pagina == "📈 Exploración de datos (EDA)":
    st.title("📈 Exploración de los datos")

    fuente_eda = st.radio(
        "Fuente de datos para explorar",
        ["Dataset limpio (post-pipeline)", "Dataset original (con nulos)"],
        horizontal=True,
    )
    df_eda = (
        datos["limpio"] if fuente_eda.startswith("Dataset limpio") else datos["original"]
    )

    if df_eda is None:
        st.warning("No se encontró el archivo correspondiente en esta carpeta.")
    else:
        st.markdown(f"**Dimensiones:** {df_eda.shape[0]} filas × {df_eda.shape[1]} columnas")

        tab1, tab2, tab3 = st.tabs(["Resumen estadístico", "Distribución por variable", "Nulos"])

        with tab1:
            st.dataframe(df_eda.describe().T, use_container_width=True)

        with tab2:
            var_sel = st.selectbox("Variable", FEATURE_COLS)
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Histograma — {var_sel}**")
                st.bar_chart(df_eda[var_sel].dropna().value_counts(bins=20).sort_index())
            with colB:
                if "Potability" in df_eda.columns:
                    st.markdown(f"**{var_sel} promedio por clase de potabilidad**")
                    st.bar_chart(df_eda.groupby("Potability")[var_sel].mean())

        with tab3:
            nulos = df_eda.isna().sum()
            nulos = nulos[nulos > 0]
            if nulos.empty:
                st.success("Este dataset no tiene valores nulos.")
            else:
                st.bar_chart(nulos)
                st.dataframe(nulos.rename("Cantidad de nulos"), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Proyecto Ingeniería y Desarrollo Sustentable · 2026")
