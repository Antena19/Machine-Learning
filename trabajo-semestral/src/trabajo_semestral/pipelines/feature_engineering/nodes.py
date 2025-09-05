"""Nodos del pipeline de ingeniería de características."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any


def extract_audio_features(main_dataset: pd.DataFrame) -> pd.DataFrame:
    """Extraer y normalizar características de audio.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        DataFrame con características de audio normalizadas
    """
    df = main_dataset.copy()
    
    # Seleccionar características de audio
    audio_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    
    # Filtrar solo las características que existen
    available_features = [col for col in audio_features if col in df.columns]
    audio_df = df[available_features].copy()
    
    # Normalizar características
    scaler = StandardScaler()
    audio_df_scaled = pd.DataFrame(
        scaler.fit_transform(audio_df),
        columns=audio_df.columns,
        index=audio_df.index
    )
    
    return audio_df_scaled


def create_popularity_features(main_dataset: pd.DataFrame) -> pd.DataFrame:
    """Crear características basadas en popularidad.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        DataFrame con características de popularidad
    """
    df = main_dataset.copy()
    
    features = pd.DataFrame(index=df.index)
    
    # Popularidad básica
    if 'popularity' in df.columns:
        features['popularity'] = df['popularity']
        features['popularity_normalized'] = df['popularity'] / 100.0
    
    # Categorías de popularidad
    if 'popularity' in df.columns:
        features['popularity_category'] = pd.cut(
            df['popularity'], 
            bins=[0, 20, 40, 60, 80, 100], 
            labels=['muy_baja', 'baja', 'media', 'alta', 'muy_alta']
        )
    
    # Popularidad de artistas (si está disponible)
    if 'artists_popularities' in df.columns:
        # Convertir string de lista a lista real y tomar el máximo
        def extract_max_popularity(pop_str):
            try:
                if isinstance(pop_str, str) and pop_str.startswith('['):
                    pop_list = eval(pop_str)
                    return max(pop_list) if pop_list else 0
                return 0
            except:
                return 0
        
        features['max_artist_popularity'] = df['artists_popularities'].apply(extract_max_popularity)
    
    return features


def create_genre_features(main_dataset: pd.DataFrame) -> pd.DataFrame:
    """Crear características basadas en géneros musicales.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        DataFrame con características de géneros
    """
    df = main_dataset.copy()
    
    features = pd.DataFrame(index=df.index)
    
    if 'artists_genres' in df.columns:
        # Extraer todos los géneros únicos
        all_genres = set()
        for genres_str in df['artists_genres'].dropna():
            if isinstance(genres_str, str) and genres_str.startswith('['):
                try:
                    genres_list = eval(genres_str)
                    all_genres.update(genres_list)
                except:
                    pass
        
        # Crear características binarias para géneros principales
        main_genres = ['pop', 'rock', 'hip hop', 'electronic', 'jazz', 'classical', 'country', 'r&b']
        
        for genre in main_genres:
            features[f'has_{genre}'] = df['artists_genres'].apply(
                lambda x: 1 if genre in str(x).lower() else 0
            )
        
        # Número total de géneros
        def count_genres(genres_str):
            try:
                if isinstance(genres_str, str) and genres_str.startswith('['):
                    genres_list = eval(genres_str)
                    return len(genres_list)
                return 0
            except:
                return 0
        
        features['num_genres'] = df['artists_genres'].apply(count_genres)
    
    return features


def create_temporal_features(main_dataset: pd.DataFrame) -> pd.DataFrame:
    """Crear características temporales.
    
    Args:
        main_dataset: Dataset principal procesado
        
    Returns:
        DataFrame con características temporales
    """
    df = main_dataset.copy()
    
    features = pd.DataFrame(index=df.index)
    
    if 'release_year' in df.columns:
        # Año de lanzamiento
        features['release_year'] = df['release_year']
        
        # Década
        features['decade'] = (df['release_year'] // 10) * 10
        
        # Edad de la canción (años desde lanzamiento)
        current_year = 2024
        features['song_age'] = current_year - df['release_year']
        
        # Categorías temporales
        features['era'] = pd.cut(
            df['release_year'],
            bins=[1900, 1980, 1990, 2000, 2010, 2020, 2030],
            labels=['pre_80s', '80s', '90s', '2000s', '2010s', '2020s']
        )
    
    if 'release_date' in df.columns:
        # Mes de lanzamiento
        features['release_month'] = pd.to_datetime(df['release_date']).dt.month
        
        # Día de la semana
        features['release_dayofweek'] = pd.to_datetime(df['release_date']).dt.dayofweek
        
        # Estación del año
        month = pd.to_datetime(df['release_date']).dt.month
        features['season'] = month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
    
    return features


def combine_features(
    audio_features: pd.DataFrame,
    popularity_features: pd.DataFrame,
    genre_features: pd.DataFrame,
    temporal_features: pd.DataFrame
) -> pd.DataFrame:
    """Combinar todas las características en un solo DataFrame.
    
    Args:
        audio_features: Características de audio
        popularity_features: Características de popularidad
        genre_features: Características de géneros
        temporal_features: Características temporales
        
    Returns:
        DataFrame combinado con todas las características
    """
    # Combinar todos los DataFrames
    combined = pd.concat([
        audio_features,
        popularity_features,
        genre_features,
        temporal_features
    ], axis=1)
    
    # Eliminar columnas duplicadas
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Rellenar valores nulos
    combined = combined.fillna(0)
    
    return combined
