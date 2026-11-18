"""
pathsnap.visualization
======================
Geospatial visualization utilities for PathSnap.
Provides interactive Folium maps for GPS trajectories, road networks,
candidate segments, ground-truth routes, and HMM/Viterbi predictions.
"""

from .map_visualizer import MapVisualizer, visualize_trip

__all__ = ["MapVisualizer", "visualize_trip"]
