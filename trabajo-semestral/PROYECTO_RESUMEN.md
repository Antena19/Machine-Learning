# Resumen del Proyecto de Recomendación Musical con Kedro

## 🎯 Objetivo
Crear un sistema completo de machine learning para recomendación musical usando datos de Spotify, implementado con pipelines modulares en Kedro.

## 📊 Datos Utilizados
- **Artistas**: Información de artistas de Spotify (popularidad, géneros, seguidores)
- **Canciones**: Características de audio de tracks (danceability, energy, valence, etc.)
- **Playlists**: Metadatos de playlists de Spotify
- **Dataset Principal**: Dataset combinado con características de audio completas

## 🏗️ Arquitectura del Pipeline

### 1. **Pipeline de Procesamiento de Datos** (`data_processing`)
- **Limpieza de datos**: Manejo de valores nulos, conversión de tipos
- **Preprocesamiento**: Normalización y estandarización de datos
- **Fusión de datasets**: Combinación de datos de diferentes fuentes

### 2. **Pipeline de Ingeniería de Características** (`feature_engineering`)
- **Características de audio**: Extracción y normalización de features de audio
- **Características de popularidad**: Métricas basadas en popularidad
- **Características de géneros**: Codificación de géneros musicales
- **Características temporales**: Features basadas en fechas de lanzamiento

### 3. **Pipeline de Entrenamiento de Modelos** (`model_training`)
- **Modelo de recomendación**: Clustering (K-Means) para agrupar canciones similares
- **Clasificador de géneros**: Random Forest para clasificar géneros musicales
- **Predictor de popularidad**: Random Forest para predecir popularidad
- **Evaluación**: Métricas de rendimiento para todos los modelos

### 4. **Pipeline de Reportes** (`reporting`)
- **Visualizaciones**: Gráficos de distribución, características y tendencias
- **Matriz de confusión**: Evaluación visual del clasificador
- **Reportes**: Documentos de rendimiento de modelos

## 🚀 Cómo Ejecutar

### Instalación
```bash
cd trabajo-semestral
pip install -r requirements.txt
```

### Ejecución Completa
```bash
kedro run
```

### Ejecución por Pipelines
```bash
# Solo procesamiento de datos
kedro run --pipeline data_processing

# Solo ingeniería de características
kedro run --pipeline feature_engineering

# Solo entrenamiento de modelos
kedro run --pipeline model_training

# Solo reportes
kedro run --pipeline reporting
```

### Visualización
```bash
kedro viz
```

## 📁 Estructura de Archivos Generados

```
data/
├── 01_raw/                    # Datos originales
├── 02_intermediate/           # Datos preprocesados
├── 03_primary/               # Datos primarios para ML
├── 04_feature/               # Características extraídas
├── 05_model_input/           # Datos de entrada para modelos
├── 06_models/                # Modelos entrenados
├── 07_model_output/          # Predicciones
└── 08_reporting/             # Visualizaciones y reportes
```

## 🔧 Características Técnicas

### Tecnologías Utilizadas
- **Kedro**: Framework de pipelines de datos
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Machine learning
- **Plotly**: Visualizaciones interactivas
- **Matplotlib/Seaborn**: Gráficos estáticos

### Modelos Implementados
1. **K-Means Clustering**: Para recomendación basada en similitud
2. **Random Forest Classifier**: Para clasificación de géneros
3. **Random Forest Regressor**: Para predicción de popularidad

### Características de Audio Utilizadas
- `danceability`: Bailabilidad
- `energy`: Energía
- `valence`: Positividad
- `tempo`: Velocidad
- `acousticness`: Acústica
- `instrumentalness`: Instrumental
- `liveness`: En vivo
- `speechiness`: Hablado
- `loudness`: Volumen

## 📈 Métricas de Evaluación

- **Clustering**: Número de clusters, balance de clusters
- **Clasificación**: Precisión (accuracy)
- **Regresión**: Error cuadrático medio (MSE), R²

## 🎨 Visualizaciones Generadas

1. **Distribución de géneros**: Gráfico de barras de géneros musicales
2. **Características de audio**: Gráfico de dispersión de features
3. **Tendencias de popularidad**: Gráfico de líneas por año
4. **Matriz de confusión**: Evaluación del clasificador
5. **Reporte de rendimiento**: Documento de métricas

## 🔄 Flujo de Datos

```
Datos Brutos → Procesamiento → Características → Modelos → Reportes
     ↓              ↓              ↓           ↓         ↓
  CSV Files → Preprocessing → Feature Eng. → Training → Viz/Reports
```

## 🛠️ Configuración

Los parámetros se configuran en `conf/base/parameters.yml`:
- División de datos (train/validation/test)
- Parámetros de modelos
- Características a utilizar
- Configuración de visualizaciones

## 📚 Documentación

- `PIPELINE_GUIDE.md`: Guía detallada de uso
- `README.md`: Documentación general del proyecto
- `PROYECTO_RESUMEN.md`: Este resumen

## 🚀 Próximos Pasos

1. **Optimización de modelos**: Ajustar hiperparámetros
2. **Nuevas características**: Agregar más features relevantes
3. **Modelos avanzados**: Implementar redes neuronales
4. **API REST**: Servir recomendaciones en tiempo real
5. **Validación cruzada**: Mejorar evaluación de modelos
6. **A/B Testing**: Probar diferentes enfoques

## 💡 Beneficios del Enfoque Kedro

- **Modularidad**: Pipelines independientes y reutilizables
- **Trazabilidad**: Seguimiento completo del flujo de datos
- **Reproducibilidad**: Configuración versionada y consistente
- **Escalabilidad**: Fácil agregar nuevos nodos y pipelines
- **Visualización**: Interfaz gráfica para explorar el pipeline
- **Testing**: Fácil testing de nodos individuales

Este proyecto demuestra las mejores prácticas de ingeniería de datos y machine learning usando Kedro como framework principal.
