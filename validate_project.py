#!/usr/bin/env python3
"""
validate_project.py

Script de validación pre-push. Verifica que todos los archivos
necesarios estén en la carpeta actual antes de hacer push a GitHub.

Uso:
  python validate_project.py

Salida:
  - Listado de archivos faltantes (si los hay)
  - Reporte de tamaños
  - Sugerencias de qué hacer
"""

from pathlib import Path
import os

# Define archivos esperados en cada categoría
ARCHIVOS_REQUERIDOS = {
    "Python (scripts)": [
        "app.py",
        "01_enriquecer_datos.py",
        "02_pipeline_limpieza.py",
        "03_entrenar_modelos.py",
    ],
    "Configuración": [
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".streamlit/config.toml",
    ],
}

ARCHIVOS_OPCIONALES = {
    "Datos (no suben a GitHub)": [
        "water_potability.csv",
        "water_potability_enriquecido.csv",
        "water_potability_limpio.csv",
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ],
    "Modelos (no suben a GitHub)": [
        "modelo_final.pkl",
        "modelo_logistic_regression.pkl",
        "modelo_random_forest.pkl",
    ],
    "Pipeline (no suben a GitHub)": [
        "scaler.pkl",
        "knn_imputer.pkl",
        "scaler_pre_imputacion.pkl",
    ],
}


def tamanio_archivo(path):
    """Retorna tamaño legible de un archivo."""
    try:
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
    except:
        return "?"
    return f"{size:.1f} TB"


def main():
    print("\n" + "=" * 70)
    print("AquaPredict — Validación de proyecto pre-push".center(70))
    print("=" * 70 + "\n")

    cwd = Path.cwd()
    print(f"📁 Carpeta actual: {cwd}\n")

    faltantes_requeridos = []
    presentes_requeridos = []

    # Verificar archivos requeridos
    print("✓ ARCHIVOS REQUERIDOS (deben estar para GitHub):")
    print("-" * 70)
    for categoria, archivos in ARCHIVOS_REQUERIDOS.items():
        print(f"\n  {categoria}:")
        for archivo in archivos:
            path = cwd / archivo
            if path.exists():
                tamanio = tamanio_archivo(path)
                print(f"    ✅ {archivo:40} ({tamanio})")
                presentes_requeridos.append(archivo)
            else:
                print(f"    ❌ {archivo:40} FALTA")
                faltantes_requeridos.append(archivo)

    # Verificar archivos opcionales (deberían existir localmente pero no subirse)
    print("\n\n⚠️  ARCHIVOS LOCALES (no suben a GitHub, pero necesarios para test):")
    print("-" * 70)
    presentes_opcionales = {}
    for categoria, archivos in ARCHIVOS_OPCIONALES.items():
        presentes_opcionales[categoria] = []
        print(f"\n  {categoria}:")
        for archivo in archivos:
            path = cwd / archivo
            if path.exists():
                tamanio = tamanio_archivo(path)
                print(f"    ✅ {archivo:40} ({tamanio})")
                presentes_opcionales[categoria].append(archivo)
            else:
                print(f"    ⚠️  {archivo:40} (opcional)")

    # Reporte final
    print("\n\n" + "=" * 70)
    print("📋 RESUMEN".center(70))
    print("=" * 70)

    total_requeridos = len(ARCHIVOS_REQUERIDOS["Python (scripts)"]) + len(
        ARCHIVOS_REQUERIDOS["Configuración"]
    )
    presentes = len(presentes_requeridos)

    print(f"\n✓ Archivos requeridos: {presentes}/{total_requeridos}")

    if faltantes_requeridos:
        print(f"\n❌ Archivos FALTANTES ({len(faltantes_requeridos)}):")
        for archivo in faltantes_requeridos:
            print(f"   - {archivo}")
        print("\n⚠️  IMPORTANTE: No puedes hacer push sin estos archivos.")
        print("   Descárgalos desde el chat de Claude y colócalos en esta carpeta.")

    total_opcionales = sum(len(v) for v in presentes_opcionales.values())
    presentes_opt = sum(len(v) for v in presentes_opcionales.values())

    print(f"\n⚠️  Archivos opcionales presentes: {presentes_opt}/{total_opcionales}")
    if presentes_opt > 0:
        print("   Estos archivos están bien guardados localmente para testing.")
        print("   ✅ El .gitignore impedirá que se suban a GitHub (correcto).")

    # Verificar .gitignore
    print("\n\n✓ VERIFICACIÓN DE .gitignore:")
    print("-" * 70)
    gitignore_path = cwd / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path) as f:
            contenido = f.read()
        if "*.csv" in contenido and "*.pkl" in contenido:
            print("  ✅ .gitignore configurado correctamente")
            print("     - No subirá archivos .csv")
            print("     - No subirá archivos .pkl")
        else:
            print("  ⚠️  .gitignore existe pero podría estar incompleto")
            print("     Asegúrate de que incluya:")
            print("       *.csv")
            print("       *.pkl")
    else:
        print("  ❌ .gitignore NO ENCONTRADO")
        print("     Sin este archivo, podrías subir datos por accidente.")

    # Recomendaciones finales
    print("\n\n" + "=" * 70)
    print("📌 RECOMENDACIONES".center(70))
    print("=" * 70 + "\n")

    if faltantes_requeridos:
        print("1. ACCIÓN INMEDIATA: Descarga los archivos faltantes")
        print("   Lugar: Outputs de la sesión de Claude en /mnt/user-data/outputs/\n")
    else:
        print("1. ✅ Todos los archivos requeridos están presentes\n")

    if presentes_opt > 0:
        print("2. ✅ Tienes archivos de datos/modelos localmente")
        print("   Esto es bueno para testing antes de hacer push\n")
    else:
        print("2. ⚠️  No tienes archivos de datos/modelos localmente")
        print("   Descárgalos desde OneDrive para que app.py funcione\n")

    print("3. PRÓXIMO PASO: Si todo está ✅, ejecuta:")
    print(f"   cd {cwd}")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'feat: AquaPredict - inicial'")
    print("   git remote add origin https://github.com/TU_USUARIO/aquapredict.git")
    print("   git push -u origin main\n")

    print("4. STREAMLIT CLOUD: Después, ve a https://share.streamlit.io")
    print("   y conecta tu repositorio. ¡Tu app estará live en ~5 min!\n")

    # Estado final
    print("=" * 70)
    if not faltantes_requeridos:
        print("✅ LISTO PARA PUSH".center(70))
        return 0
    else:
        print("❌ NO ESTÁ LISTO - FALTAN ARCHIVOS".center(70))
        return 1
    print("=" * 70 + "\n")


if __name__ == "__main__":
    exit(main())
