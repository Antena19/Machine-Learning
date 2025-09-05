"""Nodos del pipeline de procesamiento de datos."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def clean_artists_data(artists: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y procesar datos de artistas.
    
    Args:
        artists: DataFrame con datos de artistas
        
    Returns:
        DataFrame limpio de artistas
    """
    df = artists.copy()
    
    # Limpiar valores nulos
    df = df.dropna(subset=['name'])
    
    # Convertir followers a numérico
    if 'followers' in df.columns:
        df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
        df['followers'] = df['followers'].fillna(0)
    
    # Convertir popularidad a numérico
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        df['popularity'] = df['popularity'].fillna(0)
    
    # Limpiar géneros (convertir listas de strings a listas reales)
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
    return df


def clean_tracks_data(tracks: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y procesar datos de tracks.
    
    Args:
        tracks: DataFrame con datos de tracks
        
    Returns:
        DataFrame limpio de tracks
    """
    df = tracks.copy()
    
    # Limpiar valores nulos
    df = df.dropna(subset=['name'])
    
    # Convertir columnas numéricas
    numeric_columns = ['popularity', 'duration_ms', 'time_signature']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Convertir fecha de lanzamiento
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
    
    # Limpiar artistas (convertir listas de strings)
    if 'artists_names' in df.columns:
        df['artists_names'] = df['artists_names'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
    return df


def clean_playlists_data(playlists: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y procesar datos de playlists.
    
    Args:
        playlists: DataFrame con datos de playlists
        
    Returns:
        DataFrame limpio de playlists
    """
    df = playlists.copy()
    
    # Limpiar valores nulos
    df = df.dropna(subset=['name'])
    
    # Convertir columnas numéricas
    numeric_columns = ['num_tracks', 'num_followers']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    return df


def clean_main_dataset(main_dataset: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y procesar el dataset principal.
    
    Args:
        main_dataset: DataFrame principal con características de audio
        
    Returns:
        DataFrame limpio del dataset principal
    """
    df = main_dataset.copy()
    
    # Limpiar valores nulos
    df = df.dropna(subset=['name'])
    
    # Convertir características de audio a numérico
    audio_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    
    for feature in audio_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(df[feature].median())
    
    # Convertir duración a numérico
    if 'duration_ms' in df.columns:
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].median())
    
    # Convertir popularidad
    if 'popularity' in df.columns:
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        df['popularity'] = df['popularity'].fillna(0)
    
    # Convertir fecha de lanzamiento
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
    
    # Limpiar artistas
    if 'artists_names' in df.columns:
        df['artists_names'] = df['artists_names'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
    # Limpiar géneros de artistas
    if 'artists_genres' in df.columns:
        df['artists_genres'] = df['artists_genres'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
    return df


def merge_datasets(
    artists: pd.DataFrame, 
    tracks: pd.DataFrame, 
    playlists: pd.DataFrame
) -> pd.DataFrame:
    """Fusionar datasets para crear tabla de entrada del modelo.
    
    Args:
        artists: DataFrame de artistas procesados
        tracks: DataFrame de tracks procesados  
        playlists: DataFrame de playlists procesados
        
    Returns:
        DataFrame fusionado para entrada del modelo
    """
    # Por ahora, retornamos el dataset de tracks como base
    # En un caso real, aquí harías la fusión compleja de los datasets
    merged_df = tracks.copy()
    
    # Agregar información de artistas si es posible
    if 'artists_names' in merged_df.columns and 'name' in artists.columns:
        # Aquí podrías hacer un merge más sofisticado
        pass
    
    return merged_df
