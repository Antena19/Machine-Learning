"""Nodos del pipeline de entrenamiento de modelos."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Tuple, Dict, Any


def split_data(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dividir los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        features: DataFrame con características procesadas
        
    Returns:
        Tupla con (train_data, validation_data, test_data)
    """
    # Primera división: 80% entrenamiento, 20% temporal
    train_data, temp_data = train_test_split(
        features, 
        test_size=0.2, 
        random_state=42,
        stratify=features.get('popularity_category', None)
    )
    
    # Segunda división: 50% validación, 50% prueba del 20% temporal
    validation_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42,
        stratify=temp_data.get('popularity_category', None)
    )
    
    return train_data, validation_data, test_data


def train_recommendation_model(
    train_data: pd.DataFrame, 
    validation_data: pd.DataFrame
) -> Any:
    """Entrenar modelo de recomendación usando clustering.
    
    Args:
        train_data: Datos de entrenamiento
        validation_data: Datos de validación
        
    Returns:
        Modelo entrenado de recomendación
    """
    # Seleccionar características para clustering
    feature_columns = [
        'danceability', 'energy', 'valence', 'tempo', 
        'acousticness', 'instrumentalness', 'liveness'
    ]
    
    # Filtrar columnas que existen
    available_features = [col for col in feature_columns if col in train_data.columns]
    
    if not available_features:
        # Si no hay características de audio, usar todas las numéricas
        available_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_data[available_features].fillna(0)
    X_val = validation_data[available_features].fillna(0)
    
    # Entrenar modelo de clustering (K-Means)
    n_clusters = min(10, len(X_train) // 100)  # Ajustar número de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Entrenar con datos de entrenamiento
    kmeans.fit(X_train)
    
    # Validar con datos de validación
    val_clusters = kmeans.predict(X_val)
    
    print(f"Modelo de recomendación entrenado con {n_clusters} clusters")
    print(f"Distribución de clusters en validación: {np.bincount(val_clusters)}")
    
    return kmeans


def train_genre_classifier(
    train_data: pd.DataFrame, 
    validation_data: pd.DataFrame
) -> Any:
    """Entrenar clasificador de géneros musicales.
    
    Args:
        train_data: Datos de entrenamiento
        validation_data: Datos de validación
        
    Returns:
        Modelo entrenado de clasificación de géneros
    """
    # Seleccionar características
    feature_columns = [
        'danceability', 'energy', 'valence', 'tempo', 
        'acousticness', 'instrumentalness', 'liveness',
        'speechiness', 'loudness'
    ]
    
    # Filtrar columnas que existen
    available_features = [col for col in feature_columns if col in train_data.columns]
    
    if not available_features:
        available_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_data[available_features].fillna(0)
    X_val = validation_data[available_features].fillna(0)
    
    # Crear variable objetivo basada en géneros
    if 'has_pop' in train_data.columns:
        # Usar el género más común como target
        genre_cols = [col for col in train_data.columns if col.startswith('has_')]
        if genre_cols:
            y_train = train_data[genre_cols].idxmax(axis=1)
            y_val = validation_data[genre_cols].idxmax(axis=1)
        else:
            # Si no hay géneros, usar categoría de popularidad
            y_train = train_data.get('popularity_category', 'media')
            y_val = validation_data.get('popularity_category', 'media')
    else:
        # Usar categoría de popularidad como target
        y_train = train_data.get('popularity_category', 'media')
        y_val = validation_data.get('popularity_category', 'media')
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    # Entrenar modelo
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    classifier.fit(X_train, y_train_encoded)
    
    # Evaluar
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val_encoded, y_pred)
    
    print(f"Clasificador de géneros entrenado con accuracy: {accuracy:.3f}")
    
    # Guardar el encoder junto con el modelo
    model_with_encoder = {
        'model': classifier,
        'label_encoder': le
    }
    
    return model_with_encoder


def train_popularity_predictor(
    train_data: pd.DataFrame, 
    validation_data: pd.DataFrame
) -> Any:
    """Entrenar predictor de popularidad.
    
    Args:
        train_data: Datos de entrenamiento
        validation_data: Datos de validación
        
    Returns:
        Modelo entrenado de predicción de popularidad
    """
    # Seleccionar características
    feature_columns = [
        'danceability', 'energy', 'valence', 'tempo', 
        'acousticness', 'instrumentalness', 'liveness',
        'speechiness', 'loudness', 'release_year'
    ]
    
    # Filtrar columnas que existen
    available_features = [col for col in feature_columns if col in train_data.columns]
    
    if not available_features:
        available_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_data[available_features].fillna(0)
    X_val = validation_data[available_features].fillna(0)
    
    # Variable objetivo: popularidad
    if 'popularity' in train_data.columns:
        y_train = train_data['popularity']
        y_val = validation_data['popularity']
    else:
        # Si no hay popularidad, usar una variable numérica como proxy
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            y_train = train_data[numeric_cols[0]]
            y_val = validation_data[numeric_cols[0]]
        else:
            # Crear variable objetivo sintética
            y_train = np.random.randint(0, 100, len(train_data))
            y_val = np.random.randint(0, 100, len(validation_data))
    
    # Entrenar modelo
    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5
    )
    
    regressor.fit(X_train, y_train)
    
    # Evaluar
    y_pred = regressor.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"Predictor de popularidad entrenado con MSE: {mse:.3f}, R²: {r2:.3f}")
    
    return regressor


def evaluate_models(
    test_data: pd.DataFrame,
    recommendation_model: Any,
    genre_classifier: Any,
    popularity_predictor: Any
) -> pd.DataFrame:
    """Evaluar todos los modelos con datos de prueba.
    
    Args:
        test_data: Datos de prueba
        recommendation_model: Modelo de recomendación
        genre_classifier: Clasificador de géneros
        popularity_predictor: Predictor de popularidad
        
    Returns:
        DataFrame con métricas de evaluación
    """
    metrics = []
    
    # Evaluar modelo de recomendación (clustering)
    try:
        feature_columns = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'liveness'
        ]
        available_features = [col for col in feature_columns if col in test_data.columns]
        
        if not available_features:
            available_features = test_data.select_dtypes(include=[np.number]).columns.tolist()
        
        X_test = test_data[available_features].fillna(0)
        test_clusters = recommendation_model.predict(X_test)
        
        metrics.append({
            'model': 'recommendation_model',
            'metric': 'num_clusters',
            'value': len(np.unique(test_clusters)),
            'description': 'Número de clusters identificados'
        })
        
        metrics.append({
            'model': 'recommendation_model',
            'metric': 'cluster_balance',
            'value': np.std(np.bincount(test_clusters)),
            'description': 'Desviación estándar del balance de clusters'
        })
        
    except Exception as e:
        print(f"Error evaluando modelo de recomendación: {e}")
    
    # Evaluar clasificador de géneros
    try:
        if isinstance(genre_classifier, dict):
            model = genre_classifier['model']
            le = genre_classifier['label_encoder']
        else:
            model = genre_classifier
            le = None
        
        X_test = test_data[available_features].fillna(0)
        
        if le is not None:
            # Crear variable objetivo para test
            if 'has_pop' in test_data.columns:
                genre_cols = [col for col in test_data.columns if col.startswith('has_')]
                if genre_cols:
                    y_test = test_data[genre_cols].idxmax(axis=1)
                    y_test_encoded = le.transform(y_test)
                else:
                    y_test_encoded = np.zeros(len(test_data))
            else:
                y_test_encoded = np.zeros(len(test_data))
        else:
            y_test_encoded = np.zeros(len(test_data))
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        metrics.append({
            'model': 'genre_classifier',
            'metric': 'accuracy',
            'value': accuracy,
            'description': 'Precisión del clasificador'
        })
        
    except Exception as e:
        print(f"Error evaluando clasificador de géneros: {e}")
    
    # Evaluar predictor de popularidad
    try:
        y_test = test_data.get('popularity', np.random.randint(0, 100, len(test_data)))
        y_pred = popularity_predictor.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics.append({
            'model': 'popularity_predictor',
            'metric': 'mse',
            'value': mse,
            'description': 'Error cuadrático medio'
        })
        
        metrics.append({
            'model': 'popularity_predictor',
            'metric': 'r2_score',
            'value': r2,
            'description': 'Coeficiente de determinación'
        })
        
    except Exception as e:
        print(f"Error evaluando predictor de popularidad: {e}")
    
    # Crear DataFrame de métricas
    metrics_df = pd.DataFrame(metrics)
    
    print("Métricas de evaluación:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df
