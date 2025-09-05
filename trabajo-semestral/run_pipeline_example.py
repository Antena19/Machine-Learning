#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar el pipeline de recomendación musical.
Este script demuestra cómo usar los pipelines de Kedro programáticamente.
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_pipeline_example():
    """Ejecutar ejemplo del pipeline de recomendación musical."""
    
    # Configurar el proyecto
    project_path = Path(__file__).parent
    bootstrap_project(project_path)
    
    print("🎵 Iniciando Pipeline de Recomendación Musical")
    print("=" * 50)
    
    # Crear sesión de Kedro
    with KedroSession.create(project_path) as session:
        
        # 1. Ejecutar pipeline de procesamiento de datos
        print("\n📊 Ejecutando pipeline de procesamiento de datos...")
        try:
            session.run(pipeline_name="data_processing")
            print("✅ Pipeline de procesamiento de datos completado")
        except Exception as e:
            print(f"❌ Error en procesamiento de datos: {e}")
            return
        
        # 2. Ejecutar pipeline de ingeniería de características
        print("\n🔧 Ejecutando pipeline de ingeniería de características...")
        try:
            session.run(pipeline_name="feature_engineering")
            print("✅ Pipeline de ingeniería de características completado")
        except Exception as e:
            print(f"❌ Error en ingeniería de características: {e}")
            return
        
        # 3. Ejecutar pipeline de entrenamiento de modelos
        print("\n🤖 Ejecutando pipeline de entrenamiento de modelos...")
        try:
            session.run(pipeline_name="model_training")
            print("✅ Pipeline de entrenamiento de modelos completado")
        except Exception as e:
            print(f"❌ Error en entrenamiento de modelos: {e}")
            return
        
        # 4. Ejecutar pipeline de reportes
        print("\n📈 Ejecutando pipeline de reportes...")
        try:
            session.run(pipeline_name="reporting")
            print("✅ Pipeline de reportes completado")
        except Exception as e:
            print(f"❌ Error en reportes: {e}")
            return
        
        print("\n🎉 ¡Pipeline completo ejecutado exitosamente!")
        print("\nArchivos generados:")
        print("- Datos procesados en data/02_intermediate/")
        print("- Características en data/04_feature/")
        print("- Modelos en data/06_models/")
        print("- Reportes en data/08_reporting/")
        
        print("\nPara ver la visualización del pipeline:")
        print("kedro viz")


def run_individual_pipeline(pipeline_name: str):
    """Ejecutar un pipeline individual."""
    
    project_path = Path(__file__).parent
    bootstrap_project(project_path)
    
    print(f"🎵 Ejecutando pipeline: {pipeline_name}")
    print("=" * 50)
    
    with KedroSession.create(project_path) as session:
        try:
            session.run(pipeline_name=pipeline_name)
            print(f"✅ Pipeline {pipeline_name} completado exitosamente")
        except Exception as e:
            print(f"❌ Error en pipeline {pipeline_name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Ejecutar pipeline específico
        pipeline_name = sys.argv[1]
        run_individual_pipeline(pipeline_name)
    else:
        # Ejecutar pipeline completo
        run_pipeline_example()
