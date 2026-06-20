
"""
01_enriquecer_datos.py
------------------------------------------------------------
Genera un dataset "enriquecido" agregando variables ficticias
(fecha de muestra, localidad y tipo de fuente) al dataset
original de potabilidad del agua.

IMPORTANTE: Estas variables son simuladas (aleatorias) y sirven
SOLO para dar contexto narrativo al dashboard / presentación
(ej. mapas, filtros por fecha o localidad). NO deben usarse como
features del modelo de Machine Learning, ya que no tienen
relación real con la potabilidad del agua y meterlas como
predictoras introduciría ruido o relaciones espurias.

Input:  water_potability.csv
Output: water_potability_enriquecido.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------
# 0. Carpeta base: la misma carpeta donde está este script.
#    Así, sin importar en qué PC o ruta esté (C:\..., OneDrive,
#    Mac, etc.), el script siempre encuentra los archivos
#    mientras estén juntos en la misma carpeta.
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. Cargar datos originales
# ------------------------------------------------------------
df = pd.read_csv(BASE_DIR / "water_potability.csv")
n = len(df)

# ------------------------------------------------------------
# 2. Fecha de muestra (entre 2020-01-01 y hoy)
# ------------------------------------------------------------
fecha_inicio = pd.Timestamp("2020-01-01")
fecha_fin = pd.Timestamp("2026-06-20")  # fecha actual del proyecto
dias_rango = (fecha_fin - fecha_inicio).days

dias_aleatorios = rng.integers(0, dias_rango, size=n)
df["fecha_muestra"] = fecha_inicio + pd.to_timedelta(dias_aleatorios, unit="D")
df["fecha_muestra"] = df["fecha_muestra"].dt.strftime("%Y-%m-%d")

# ------------------------------------------------------------
# 3. Localidad (Chile) — comunas/localidades de distintas regiones
# ------------------------------------------------------------
localidades_chile = [
    "Arica", "Iquique", "Antofagasta", "Calama", "Copiapó", "La Serena",
    "Coquimbo", "Ovalle", "Valparaíso", "Viña del Mar", "San Antonio",
    "Quillota", "Santiago", "Puente Alto", "Maipú", "San Bernardo",
    "Rancagua", "San Fernando", "Talca", "Curicó", "Linares",
    "Chillán", "Los Ángeles", "Concepción", "Talcahuano", "Coronel",
    "Temuco", "Villarrica", "Angol", "Valdivia", "Osorno",
    "Puerto Montt", "Castro", "Coyhaique", "Punta Arenas"
]

regiones_por_localidad = {
    "Arica": "Arica y Parinacota", "Iquique": "Tarapacá",
    "Antofagasta": "Antofagasta", "Calama": "Antofagasta",
    "Copiapó": "Atacama", "La Serena": "Coquimbo", "Coquimbo": "Coquimbo",
    "Ovalle": "Coquimbo", "Valparaíso": "Valparaíso",
    "Viña del Mar": "Valparaíso", "San Antonio": "Valparaíso",
    "Quillota": "Valparaíso", "Santiago": "Metropolitana",
    "Puente Alto": "Metropolitana", "Maipú": "Metropolitana",
    "San Bernardo": "Metropolitana", "Rancagua": "O'Higgins",
    "San Fernando": "O'Higgins", "Talca": "Maule", "Curicó": "Maule",
    "Linares": "Maule", "Chillán": "Ñuble", "Los Ángeles": "Biobío",
    "Concepción": "Biobío", "Talcahuano": "Biobío", "Coronel": "Biobío",
    "Temuco": "Araucanía", "Villarrica": "Araucanía", "Angol": "Araucanía",
    "Valdivia": "Los Ríos", "Osorno": "Los Lagos",
    "Puerto Montt": "Los Lagos", "Castro": "Los Lagos",
    "Coyhaique": "Aysén", "Punta Arenas": "Magallanes"
}

df["localidad"] = rng.choice(localidades_chile, size=n)
df["region"] = df["localidad"].map(regiones_por_localidad)

# ------------------------------------------------------------
# 4. Tipo de fuente de agua
# ------------------------------------------------------------
tipos_fuente = ["Red urbana", "Pozo", "Río", "Estero/Vertiente", "Embalse"]
# Probabilidades no uniformes (más realista: la mayoría es red urbana)
probs_fuente = [0.45, 0.20, 0.15, 0.10, 0.10]
df["tipo_fuente"] = rng.choice(tipos_fuente, size=n, p=probs_fuente)

# ------------------------------------------------------------
# 5. ID de muestra (identificador único legible)
# ------------------------------------------------------------
df.insert(0, "id_muestra", [f"M-{i+1:05d}" for i in range(n)])

# ------------------------------------------------------------
# 6. Guardar
# ------------------------------------------------------------
df.to_csv(BASE_DIR / "water_potability_enriquecido.csv", index=False)

print("Dataset enriquecido generado correctamente.")
print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
print(df[["id_muestra", "fecha_muestra", "localidad", "region", "tipo_fuente"]].head())

