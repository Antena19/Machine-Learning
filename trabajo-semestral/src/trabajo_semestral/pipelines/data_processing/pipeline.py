"""Pipeline de procesamiento de datos para limpiar y preparar los datos de Spotify."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_artists_data,
    clean_tracks_data,
    clean_playlists_data,
    clean_main_dataset,
    merge_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear el pipeline de procesamiento de datos.

    Returns:
        Pipeline: Pipeline de Kedro para procesamiento de datos.
    """
    return pipeline(
        [
            # Limpieza de datos individuales
            node(
                func=clean_artists_data,
                inputs="artists",
                outputs="preprocessed_artists",
                name="clean_artists_data",
                tags=["data_cleaning", "artists"],
            ),
            node(
                func=clean_tracks_data,
                inputs="tracks",
                outputs="preprocessed_tracks",
                name="clean_tracks_data",
                tags=["data_cleaning", "tracks"],
            ),
            node(
                func=clean_playlists_data,
                inputs="playlists",
                outputs="preprocessed_playlists",
                name="clean_playlists_data",
                tags=["data_cleaning", "playlists"],
            ),
            node(
                func=clean_main_dataset,
                inputs="main_dataset",
                outputs="preprocessed_main_dataset",
                name="clean_main_dataset",
                tags=["data_cleaning", "main_dataset"],
            ),
            # Fusión de datasets
            node(
                func=merge_datasets,
                inputs=["preprocessed_artists", "preprocessed_tracks", "preprocessed_playlists"],
                outputs="model_input_table",
                name="merge_datasets",
                tags=["data_merging", "model_input"],
            ),
        ],
        tags="data_processing",
    )
