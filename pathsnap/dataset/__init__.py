"""
pathsnap.dataset
================
Dataset construction, metadata management, and group-aware splitting.
"""

from .dataset_builder import (
    DatasetBuilder,
    get_feature_matrix,
    get_target_vector,
    get_groups,
    METADATA_COLUMNS,
    TARGET_COLUMN,
)

__all__ = [
    "DatasetBuilder",
    "get_feature_matrix",
    "get_target_vector",
    "get_groups",
    "METADATA_COLUMNS",
    "TARGET_COLUMN",
]
