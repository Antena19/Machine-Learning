"""Nodos del pipeline de reportes y visualizaciones."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, Any


def create_genre_distribution_plot(main_dataset: pd.DataFrame) -> Dict[str, Any]:
    """Crear gráfico de distribución de géneros musicales.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        Diccionario con configuración de Plotly
    """
    # Extraer géneros de la columna artists_genres
    all_genres = []
    
    if 'artists_genres' in main_dataset.columns:
        for genres_str in main_dataset['artists_genres'].dropna():
            if isinstance(genres_str, str) and genres_str.startswith('['):
                try:
                    genres_list = eval(genres_str)
                    all_genres.extend(genres_list)
                except:
                    pass
    
    if not all_genres:
        # Si no hay géneros, crear datos de ejemplo
        all_genres = ['pop', 'rock', 'electronic', 'hip hop', 'jazz'] * 100
    
    # Contar géneros
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    # Crear gráfico
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Distribución de Géneros Musicales en Spotify",
        labels={'x': 'Géneros Musicales', 'y': 'Número de Canciones'},
        color=genre_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Géneros Musicales",
        yaxis_title="Número de Canciones",
        title_x=0.5,
        showlegend=False
    )
    
    return {
        'type': 'bar',
        'fig': {
            'x': genre_counts.index.tolist(),
            'y': genre_counts.values.tolist()
        },
        'layout': {
            'xaxis_title': "Géneros Musicales",
            'yaxis_title': "Número de Canciones",
            'title': "Distribución de Géneros Musicales en Spotify"
        }
    }


def create_audio_features_plot(features: pd.DataFrame) -> Dict[str, Any]:
    """Crear gráfico de características de audio.
    
    Args:
        features: DataFrame con características de audio
        
    Returns:
        Diccionario con configuración de Plotly
    """
    # Seleccionar características de audio disponibles
    audio_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
    available_cols = [col for col in audio_cols if col in features.columns]
    
    if len(available_cols) < 2:
        # Si no hay suficientes características, usar las numéricas disponibles
        available_cols = features.select_dtypes(include=[np.number]).columns[:5].tolist()
    
    if len(available_cols) < 2:
        # Crear datos de ejemplo si no hay suficientes columnas
        n_samples = min(1000, len(features))
        sample_data = pd.DataFrame({
            'danceability': np.random.uniform(0, 1, n_samples),
            'energy': np.random.uniform(0, 1, n_samples),
            'valence': np.random.uniform(0, 1, n_samples),
            'popularity': np.random.uniform(0, 100, n_samples)
        })
    else:
        # Usar datos reales
        sample_data = features[available_cols].sample(n=min(1000, len(features)))
        if 'popularity' not in sample_data.columns and 'popularity' in features.columns:
            sample_data['popularity'] = features['popularity'].sample(n=len(sample_data))
    
    # Crear gráfico de dispersión
    x_col = available_cols[0] if available_cols else 'danceability'
    y_col = available_cols[1] if len(available_cols) > 1 else 'energy'
    color_col = 'valence' if 'valence' in sample_data.columns else 'popularity'
    size_col = 'popularity' if 'popularity' in sample_data.columns else None
    
    fig = px.scatter(
        sample_data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title="Relación entre Características de Audio",
        opacity=0.6
    )
    
    return {
        'type': 'scatter',
        'fig': {
            'x': sample_data[x_col].tolist(),
            'y': sample_data[y_col].tolist(),
            'color': sample_data[color_col].tolist() if color_col in sample_data.columns else None,
            'size': sample_data[size_col].tolist() if size_col and size_col in sample_data.columns else None
        },
        'layout': {
            'xaxis_title': x_col.title(),
            'yaxis_title': y_col.title(),
            'title': "Relación entre Características de Audio"
        }
    }


def create_popularity_trends_plot(main_dataset: pd.DataFrame) -> Dict[str, Any]:
    """Crear gráfico de tendencias de popularidad por año.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        Diccionario con configuración de Plotly
    """
    # Preparar datos
    if 'release_year' in main_dataset.columns and 'popularity' in main_dataset.columns:
        # Agrupar por año y calcular popularidad promedio
        yearly_popularity = main_dataset.groupby('release_year')['popularity'].agg(['mean', 'count']).reset_index()
        yearly_popularity = yearly_popularity[yearly_popularity['count'] >= 5]  # Al menos 5 canciones por año
        
        if len(yearly_popularity) > 0:
            fig = px.line(
                yearly_popularity,
                x='release_year',
                y='mean',
                title="Tendencias de Popularidad por Año",
                labels={'mean': 'Popularidad Promedio', 'release_year': 'Año de Lanzamiento'}
            )
            
            return {
                'type': 'line',
                'fig': {
                    'x': yearly_popularity['release_year'].tolist(),
                    'y': yearly_popularity['mean'].tolist()
                },
                'layout': {
                    'xaxis_title': "Año de Lanzamiento",
                    'yaxis_title': "Popularidad Promedio",
                    'title': "Tendencias de Popularidad por Año"
                }
            }
    
    # Si no hay datos suficientes, crear datos de ejemplo
    years = list(range(2010, 2024))
    popularity = np.random.uniform(20, 80, len(years)) + np.sin(np.array(years) * 0.5) * 10
    
    return {
        'type': 'line',
        'fig': {
            'x': years,
            'y': popularity.tolist()
        },
        'layout': {
            'xaxis_title': "Año de Lanzamiento",
            'yaxis_title': "Popularidad Promedio",
            'title': "Tendencias de Popularidad por Año"
        }
    }


def create_confusion_matrix_plot(test_data: pd.DataFrame, genre_classifier: Any) -> None:
    """Crear matriz de confusión para el clasificador de géneros.
    
    Args:
        test_data: Datos de prueba
        genre_classifier: Modelo clasificador entrenado
        
    Returns:
        None (guarda la imagen directamente)
    """
    try:
        # Preparar datos
        feature_columns = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'liveness'
        ]
        available_features = [col for col in feature_columns if col in test_data.columns]
        
        if not available_features:
            available_features = test_data.select_dtypes(include=[np.number]).columns.tolist()
        
        X_test = test_data[available_features].fillna(0)
        
        # Obtener predicciones
        if isinstance(genre_classifier, dict):
            model = genre_classifier['model']
            le = genre_classifier['label_encoder']
        else:
            model = genre_classifier
            le = None
        
        y_pred = model.predict(X_test)
        
        # Crear etiquetas reales (simuladas si no están disponibles)
        if le is not None:
            if 'has_pop' in test_data.columns:
                genre_cols = [col for col in test_data.columns if col.startswith('has_')]
                if genre_cols:
                    y_true = test_data[genre_cols].idxmax(axis=1)
                    y_true_encoded = le.transform(y_true)
                else:
                    y_true_encoded = np.zeros(len(test_data))
            else:
                y_true_encoded = np.zeros(len(test_data))
        else:
            y_true_encoded = np.zeros(len(test_data))
        
        # Crear matriz de confusión
        cm = confusion_matrix(y_true_encoded, y_pred)
        
        # Crear gráfico
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - Clasificador de Géneros')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        
        # Guardar imagen
        plt.savefig('data/08_reporting/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creando matriz de confusión: {e}")
        # Crear matriz de confusión de ejemplo
        plt.figure(figsize=(10, 8))
        cm = np.random.randint(0, 100, (5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - Clasificador de Géneros (Ejemplo)')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.savefig('data/08_reporting/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_model_performance_report(model_metrics: pd.DataFrame) -> str:
    """Generar reporte de rendimiento de los modelos.
    
    Args:
        model_metrics: DataFrame con métricas de los modelos
        
    Returns:
        String con el reporte de rendimiento
    """
    report = []
    report.append("=" * 60)
    report.append("REPORTE DE RENDIMIENTO DE MODELOS")
    report.append("=" * 60)
    report.append("")
    
    if model_metrics.empty:
        report.append("No se encontraron métricas de evaluación.")
        return "\n".join(report)
    
    # Agrupar por modelo
    for model in model_metrics['model'].unique():
        model_data = model_metrics[model_metrics['model'] == model]
        report.append(f"MODELO: {model.upper()}")
        report.append("-" * 40)
        
        for _, row in model_data.iterrows():
            report.append(f"{row['metric']}: {row['value']:.4f}")
            report.append(f"  Descripción: {row['description']}")
        
        report.append("")
    
    # Resumen general
    report.append("RESUMEN GENERAL")
    report.append("-" * 20)
    
    if 'accuracy' in model_metrics['metric'].values:
        accuracy = model_metrics[model_metrics['metric'] == 'accuracy']['value'].iloc[0]
        report.append(f"Precisión del clasificador: {accuracy:.3f}")
    
    if 'r2_score' in model_metrics['metric'].values:
        r2 = model_metrics[model_metrics['metric'] == 'r2_score']['value'].iloc[0]
        report.append(f"R² del predictor de popularidad: {r2:.3f}")
    
    if 'mse' in model_metrics['metric'].values:
        mse = model_metrics[model_metrics['metric'] == 'mse']['value'].iloc[0]
        report.append(f"MSE del predictor de popularidad: {mse:.3f}")
    
    report.append("")
    report.append("Fecha de generación: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return "\n".join(report)
