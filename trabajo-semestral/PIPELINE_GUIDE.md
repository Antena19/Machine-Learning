# Guía de Pipelines de Kedro - Proyecto de Recomendación Musical

Este proyecto implementa un sistema completo de machine learning para recomendación musical usando datos de Spotify, organizado en pipelines modulares con Kedro.

## Estructura de Pipelines

### 1. Pipeline de Procesamiento de Datos (`data_processing`)
**Propósito**: Limpiar y preparar los datos brutos de Spotify.

**Nodos**:
- `clean_artists_data`: Limpia datos de artistas
- `clean_tracks_data`: Limpia datos de canciones
- `clean_playlists_data`: Limpia datos de playlists
- `clean_main_dataset`: Limpia el dataset principal
- `merge_datasets`: Fusiona datasets para crear tabla de entrada

**Entradas**: `artists`, `tracks`, `playlists`, `main_dataset`
**Salidas**: `preprocessed_artists`, `preprocessed_tracks`, `preprocessed_playlists`, `preprocessed_main_dataset`, `model_input_table`

### 2. Pipeline de Ingeniería de Características (`feature_engineering`)
**Propósito**: Extraer y crear características relevantes para los modelos de ML.

**Nodos**:
- `extract_audio_features`: Extrae y normaliza características de audio
- `create_popularity_features`: Crea características basadas en popularidad
- `create_genre_features`: Crea características de géneros musicales
- `create_temporal_features`: Crea características temporales
- `combine_features`: Combina todas las características

**Entradas**: `preprocessed_main_dataset`
**Salidas**: `audio_features`, `popularity_features`, `genre_features`, `temporal_features`, `audio_features_engineered`

### 3. Pipeline de Entrenamiento de Modelos (`model_training`)
**Propósito**: Entrenar modelos de machine learning para diferentes tareas.

**Nodos**:
- `split_data`: Divide datos en entrenamiento, validación y prueba
- `train_recommendation_model`: Entrena modelo de recomendación (clustering)
- `train_genre_classifier`: Entrena clasificador de géneros
- `train_popularity_predictor`: Entrena predictor de popularidad
- `evaluate_models`: Evalúa rendimiento de todos los modelos

**Entradas**: `audio_features_engineered`
**Salidas**: `train_data`, `validation_data`, `test_data`, `recommendation_model`, `genre_classifier`, `popularity_predictor`, `model_metrics`

### 4. Pipeline de Reportes (`reporting`)
**Propósito**: Generar visualizaciones y reportes de los resultados.

**Nodos**:
- `create_genre_distribution_plot`: Gráfico de distribución de géneros
- `create_audio_features_plot`: Gráfico de características de audio
- `create_popularity_trends_plot`: Gráfico de tendencias de popularidad
- `create_confusion_matrix_plot`: Matriz de confusión del clasificador
- `generate_model_performance_report`: Reporte de rendimiento

**Entradas**: `preprocessed_main_dataset`, `audio_features_engineered`, `test_data`, `genre_classifier`, `model_metrics`
**Salidas**: `genre_distribution_plot`, `audio_features_plot`, `popularity_trends_plot`, `confusion_matrix_plot`, `model_performance_report`

## Cómo Ejecutar los Pipelines

### Ejecutar Pipeline Completo
```bash
kedro run
```

### Ejecutar Pipelines Individuales
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

### Ejecutar Nodos Específicos
```bash
# Solo limpiar datos principales
kedro run --node clean_main_dataset

# Solo entrenar modelo de recomendación
kedro run --node train_recommendation_model

# Solo crear visualizaciones
kedro run --node create_genre_distribution_plot
```

### Ejecutar con Tags
```bash
# Ejecutar solo nodos de limpieza de datos
kedro run --tag data_cleaning

# Ejecutar solo nodos de entrenamiento de modelos
kedro run --tag model_training

# Ejecutar solo visualizaciones
kedro run --tag visualization
```

## Visualizar el Pipeline

Para ver la estructura completa del pipeline:

```bash
kedro viz
```

Esto abrirá una interfaz web donde puedes:
- Ver la estructura completa del pipeline
- Explorar las dependencias entre nodos
- Ver los datos que fluyen entre nodos
- Ejecutar nodos individuales

## Configuración

Los parámetros del proyecto se encuentran en `conf/base/parameters.yml`:

- **División de datos**: Configuración para train/validation/test split
- **Modelos**: Parámetros para cada modelo de ML
- **Características**: Lista de características de audio a usar
- **Géneros**: Géneros musicales principales para clasificación
- **Visualizaciones**: Configuración de gráficos y reportes

## Estructura de Datos

### Datos de Entrada (01_raw)
- `artists.csv`: Información de artistas
- `tracks.csv`: Información de canciones
- `playlists.csv`: Información de playlists
- `main_dataset.csv`: Dataset principal con características de audio

### Datos Procesados (02_intermediate, 03_primary, 04_feature)
- Datos limpios y preprocesados
- Características extraídas y normalizadas
- Datos preparados para machine learning

### Modelos (06_models)
- `recommendation_model.pickle`: Modelo de recomendación
- `genre_classifier.pickle`: Clasificador de géneros
- `popularity_predictor.pickle`: Predictor de popularidad

### Salidas (07_model_output, 08_reporting)
- Predicciones de los modelos
- Métricas de evaluación
- Visualizaciones y reportes

## Troubleshooting

### Errores Comunes

1. **Error de memoria**: Si el dataset es muy grande, considera usar muestreo:
   ```python
   # En los nodos, agregar:
   df = df.sample(n=min(10000, len(df)))
   ```

2. **Columnas faltantes**: Los nodos están diseñados para manejar columnas faltantes automáticamente.

3. **Datos nulos**: Los nodos incluyen manejo robusto de valores nulos.

### Logs y Debugging

Los logs se guardan en `info.log`. Para más información de debugging:

```bash
kedro run --verbose
```

## Próximos Pasos

1. **Optimización**: Ajustar parámetros de modelos en `parameters.yml`
2. **Nuevas características**: Agregar más características en el pipeline de feature engineering
3. **Nuevos modelos**: Implementar modelos más avanzados (neural networks, etc.)
4. **Validación cruzada**: Agregar validación cruzada para mejor evaluación
5. **API**: Crear API REST para servir recomendaciones en tiempo real
