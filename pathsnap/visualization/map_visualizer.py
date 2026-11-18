"""
pathsnap.visualization.map_visualizer
=====================================
Interactive Folium-based geographical visualization for PathSnap.

Renders:
1. Raw GPS observations (.track) with speed/heading popups and kinematic gap indicators.
2. Local road-network geometry (.nodes, .arcs) filtered within the trajectory bounding box.
3. Spatial candidate road segments retrieved via the KD-tree query.
4. Reference ground-truth route (.route) for permitted training/development trips.
5. Predicted road segments (HMM / Viterbi decoded path) when available.
6. Multi-basemap support (OSM, CartoDB Positron, CartoDB Dark Matter), layer controls,
   and a custom floating HTML legend.
7. Automated high-resolution screenshot export via headless Chrome/Edge.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

import folium
from folium import plugins
import branca.element

from ..preprocessing.gps_preprocessor import GPSPreprocessor, GPSTrajectory
from ..preprocessing.network_preprocessor import NetworkPreprocessor, RoadNetwork
from ..candidate_generation.spatial_index import SampledArcKDTree, CandidateMatch


# Distinguishable Color Palette
COLOR_GPS_NORMAL = "#0288D1"       # Electric Blue
COLOR_GPS_GAP = "#E91E63"          # Vivid Pink/Magenta (for large gaps/jumps)
COLOR_GPS_LINE = "#03A9F4"         # Light Blue dashed line
COLOR_ROAD_NETWORK = "#78909C"     # Slate Gray (infrastructure)
COLOR_CANDIDATE_ARC = "#FFA000"    # Amber / Goldenrod (KD-Tree candidates)
COLOR_GROUND_TRUTH = "#2E7D32"     # Emerald Forest Green (Ground-truth route)
COLOR_PREDICTED_ROUTE = "#7B1FA2"  # Deep Purple / Violet (Predicted route)


class MapVisualizer:
    """
    Constructs multi-layered interactive Folium maps for GPS map-matching analysis.
    """

    def __init__(
        self,
        default_tiles: str = "OpenStreetMap",
        buffer_deg: float = 0.005,
        max_network_arcs: int = 5000,
    ):
        self.default_tiles = default_tiles
        self.buffer_deg = buffer_deg
        self.max_network_arcs = max_network_arcs

    def create_trip_map(
        self,
        trajectory: GPSTrajectory,
        network: RoadNetwork,
        candidate_matches: Optional[List[List[CandidateMatch]]] = None,
        reference_route_arcs: Optional[List[int]] = None,
        predicted_route_arcs: Optional[List[int]] = None,
        title: Optional[str] = None,
    ) -> folium.Map:
        """
        Builds a comprehensive Folium map containing all layers.

        Parameters
        ----------
        trajectory : GPSTrajectory
            Preprocessed GPS trajectory.
        network : RoadNetwork
            Preprocessed road network.
        candidate_matches : list of list of CandidateMatch, optional
            Queried candidates per observation.
        reference_route_arcs : list of int, optional
            Reference ground-truth sequence of arc IDs.
        predicted_route_arcs : list of int, optional
            Predicted sequence of arc IDs from HMM / Viterbi decoding.
        title : str, optional
            Map title displayed in legend and header.

        Returns
        -------
        folium.Map
        """
        df_gps = trajectory.df
        trip_id = trajectory.trip_id
        if title is None:
            title = f"PathSnap Map Matching — Trip {trip_id}"

        # 1. Compute trajectory bounding box with buffer
        min_lat = float(df_gps["lat"].min()) - self.buffer_deg
        max_lat = float(df_gps["lat"].max()) + self.buffer_deg
        min_lon = float(df_gps["lon"].min()) - self.buffer_deg
        max_lon = float(df_gps["lon"].max()) + self.buffer_deg
        center_lat = float(df_gps["lat"].mean())
        center_lon = float(df_gps["lon"].mean())

        # 2. Initialize Folium Map with multiple basemaps
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles=None,
            control_scale=True,
        )

        # Add base tile layers
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap (Standard)", max_zoom=19).add_to(m)
        folium.TileLayer("CartoDB positron", name="CartoDB Positron (Clean Light)", max_zoom=19).add_to(m)
        folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark Matter (High Contrast)", max_zoom=19).add_to(m)

        # 3. Layer: Local Road Network Infrastructure
        fg_network = folium.FeatureGroup(name="1. Local Road Network", show=True)
        nodes_lonlat = network.nodes_lonlat
        arcs = network.arcs
        arc_lengths = network.arc_lengths
        num_arcs = len(arcs)

        # Filter arcs within bounding box
        arc_count = 0
        for arc_id in range(num_arcs):
            u, v = arcs[arc_id]
            lon_u, lat_u = nodes_lonlat[u]
            lon_v, lat_v = nodes_lonlat[v]

            # Check if either endpoint lies in bounding box
            in_bbox = (
                (min_lat <= lat_u <= max_lat and min_lon <= lon_u <= max_lon) or
                (min_lat <= lat_v <= max_lat and min_lon <= lon_v <= max_lon)
            )
            if in_bbox:
                length_m = float(arc_lengths[arc_id])
                popup_html = (
                    f"<b>Road Arc #{arc_id}</b><br>"
                    f"From Node: {u} &rarr; To Node: {v}<br>"
                    f"Length: {length_m:.1f} m"
                )
                folium.PolyLine(
                    locations=[[lat_u, lon_u], [lat_v, lon_v]],
                    color=COLOR_ROAD_NETWORK,
                    weight=1.8,
                    opacity=0.45,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"Arc #{arc_id} ({length_m:.0f}m)",
                ).add_to(fg_network)
                arc_count += 1
                if arc_count >= self.max_network_arcs:
                    break

        fg_network.add_to(m)

        # 4. Layer: Spatial Candidates (KD-Tree Query)
        fg_candidates = folium.FeatureGroup(name="2. Spatial Candidates (KD-Tree Query)", show=True)
        if candidate_matches is not None:
            unique_cand_arc_ids = set()
            for cands in candidate_matches:
                for c in cands:
                    unique_cand_arc_ids.add(c.arc_id)

            for arc_id in sorted(unique_cand_arc_ids):
                u, v = arcs[arc_id]
                lon_u, lat_u = nodes_lonlat[u]
                lon_v, lat_v = nodes_lonlat[v]
                length_m = float(arc_lengths[arc_id])
                popup_html = (
                    f"<b>Candidate Arc #{arc_id}</b><br>"
                    f"Endpoints: {u} &rarr; {v}<br>"
                    f"Length: {length_m:.1f} m"
                )
                folium.PolyLine(
                    locations=[[lat_u, lon_u], [lat_v, lon_v]],
                    color=COLOR_CANDIDATE_ARC,
                    weight=3.2,
                    opacity=0.65,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"Candidate Arc #{arc_id}",
                ).add_to(fg_candidates)

        fg_candidates.add_to(m)

        # 5. Layer: Reference Ground-Truth Route (.route)
        fg_route = folium.FeatureGroup(name="3. Reference Ground-Truth Route (.route)", show=True)
        if reference_route_arcs is not None and len(reference_route_arcs) > 0:
            route_coords = []
            for step_idx, arc_id in enumerate(reference_route_arcs):
                u, v = arcs[arc_id]
                lon_u, lat_u = nodes_lonlat[u]
                lon_v, lat_v = nodes_lonlat[v]
                length_m = float(arc_lengths[arc_id])

                folium.PolyLine(
                    locations=[[lat_u, lon_u], [lat_v, lon_v]],
                    color=COLOR_GROUND_TRUTH,
                    weight=5.5,
                    opacity=0.90,
                    popup=folium.Popup(f"<b>Ground-Truth Route Step #{step_idx}</b><br>Arc #{arc_id} ({length_m:.1f}m)", max_width=250),
                    tooltip=f"GT Step #{step_idx}: Arc #{arc_id}",
                ).add_to(fg_route)

                route_coords.append([lat_u, lon_u])
                if step_idx == len(reference_route_arcs) - 1:
                    route_coords.append([lat_v, lon_v])

            # Start and End Markers for Ground-Truth Route
            if len(route_coords) >= 2:
                folium.Marker(
                    location=route_coords[0],
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                    popup="<b>Route Start</b> (Ground Truth)",
                    tooltip="Route Start (GT)",
                ).add_to(fg_route)
                folium.Marker(
                    location=route_coords[-1],
                    icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
                    popup="<b>Route Destination</b> (Ground Truth)",
                    tooltip="Route Destination (GT)",
                ).add_to(fg_route)

        fg_route.add_to(m)

        # 6. Layer: Predicted Route (HMM / Viterbi Decoded)
        fg_pred = folium.FeatureGroup(name="4. Predicted Route (HMM / Viterbi)", show=True)
        if predicted_route_arcs is not None and len(predicted_route_arcs) > 0:
            pred_coords = []
            for p_step, arc_id in enumerate(predicted_route_arcs):
                u, v = arcs[arc_id]
                lon_u, lat_u = nodes_lonlat[u]
                lon_v, lat_v = nodes_lonlat[v]
                length_m = float(arc_lengths[arc_id])

                folium.PolyLine(
                    locations=[[lat_u, lon_u], [lat_v, lon_v]],
                    color=COLOR_PREDICTED_ROUTE,
                    weight=4.5,
                    opacity=0.95,
                    dash_array="8, 6",
                    popup=folium.Popup(f"<b>Predicted Route Step #{p_step}</b><br>Arc #{arc_id} ({length_m:.1f}m)", max_width=250),
                    tooltip=f"Predicted Step #{p_step}: Arc #{arc_id}",
                ).add_to(fg_pred)

                pred_coords.append([lat_u, lon_u])
                if p_step == len(predicted_route_arcs) - 1:
                    pred_coords.append([lat_v, lon_v])

            if len(pred_coords) >= 2:
                folium.Marker(
                    location=pred_coords[0],
                    icon=folium.Icon(color="purple", icon="rocket", prefix="fa"),
                    popup="<b>Predicted Route Origin</b>",
                    tooltip="Predicted Origin",
                ).add_to(fg_pred)
                folium.Marker(
                    location=pred_coords[-1],
                    icon=folium.Icon(color="darkpurple", icon="trophy", prefix="fa"),
                    popup="<b>Predicted Route Destination</b>",
                    tooltip="Predicted Destination",
                ).add_to(fg_pred)

        fg_pred.add_to(m)

        # 7. Layer: Raw GPS Observations (.track)
        fg_gps = folium.FeatureGroup(name="5. Raw GPS Observations (.track)", show=True)
        gps_latlons = []

        for i, row in df_gps.iterrows():
            lat = float(row["lat"])
            lon = float(row["lon"])
            ts = float(row["timestamp"])
            speed_mps = float(row.get("speed_mps", 0.0))
            speed_kmh = speed_mps * 3.6
            heading = float(row.get("bearing_rad", 0.0)) * 180.0 / np.pi
            is_gap = bool(row.get("is_gap", False) or row.get("is_large_gap", False) or row.get("is_jump", False))

            gps_latlons.append([lat, lon])

            marker_color = COLOR_GPS_GAP if is_gap else COLOR_GPS_NORMAL
            fill_color = "#FF80AB" if is_gap else "#81D4FA"

            gap_status = "<span style='color:red;font-weight:bold;'>ANOMALOUS GAP/JUMP</span>" if is_gap else "<span style='color:green;'>Normal Kinematics</span>"

            popup_content = (
                f"<div style='font-family:Arial, sans-serif; font-size:12px; min-width:180px;'>"
                f"<h4 style='margin:0 0 5px 0; color:#0288D1;'>GPS Observation #{i}</h4>"
                f"<b>Timestamp:</b> {ts:.1f} s<br>"
                f"<b>Latitude:</b> {lat:.6f}&deg;<br>"
                f"<b>Longitude:</b> {lon:.6f}&deg;<br>"
                f"<b>Speed:</b> {speed_mps:.2f} m/s ({speed_kmh:.1f} km/h)<br>"
                f"<b>Heading:</b> {heading:.1f}&deg;<br>"
                f"<b>Status:</b> {gap_status}"
                f"</div>"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=marker_color,
                weight=1.5,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"Obs #{i} | {speed_kmh:.1f} km/h",
            ).add_to(fg_gps)

        # Connect GPS points with thin dashed line
        if len(gps_latlons) > 1:
            folium.PolyLine(
                locations=gps_latlons,
                color=COLOR_GPS_LINE,
                weight=1.8,
                opacity=0.6,
                dash_array="4, 6",
            ).add_to(fg_gps)

        # GPS Start and Finish pins
        if len(gps_latlons) >= 2:
            folium.Marker(
                location=gps_latlons[0],
                icon=folium.Icon(color="blue", icon="location-arrow", prefix="fa"),
                popup=f"<b>GPS Trajectory Start</b><br>Obs #0 (t={df_gps.iloc[0]['timestamp']:.1f}s)",
                tooltip="GPS Start",
            ).add_to(fg_gps)
            folium.Marker(
                location=gps_latlons[-1],
                icon=folium.Icon(color="darkblue", icon="bullseye", prefix="fa"),
                popup=f"<b>GPS Trajectory End</b><br>Obs #{len(df_gps)-1} (t={df_gps.iloc[-1]['timestamp']:.1f}s)",
                tooltip="GPS End",
            ).add_to(fg_gps)

        fg_gps.add_to(m)

        # 8. Add Floating HTML Legend
        has_gt = reference_route_arcs is not None and len(reference_route_arcs) > 0
        has_pred = predicted_route_arcs is not None and len(predicted_route_arcs) > 0

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 290px;
            background-color: white;
            border: 2px solid #37474F;
            border-radius: 8px;
            z-index: 9999;
            font-size: 12px;
            font-family: Arial, sans-serif;
            padding: 12px 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            opacity: 0.95;
            line-height: 1.5;
        ">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 8px; color: #263238; border-bottom: 1px solid #ECEFF1; padding-bottom: 4px;">
                🗺️ {title}
            </div>
            <div style="display:flex; align-items:center; margin-bottom: 5px;">
                <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:{COLOR_GPS_NORMAL}; border:1.5px solid white; margin-right:8px; box-shadow:0 0 2px black;"></span>
                <span><b>Raw GPS Observations</b> ({len(df_gps):,} pts)</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom: 5px;">
                <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:{COLOR_GPS_GAP}; border:1.5px solid white; margin-right:8px; box-shadow:0 0 2px black;"></span>
                <span><b>GPS Gap / Kinematic Jump</b></span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom: 5px;">
                <span style="display:inline-block; width:22px; height:3px; background:{COLOR_ROAD_NETWORK}; margin-right:8px;"></span>
                <span><b>Local Road Network</b> ({arc_count:,} arcs)</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom: 5px;">
                <span style="display:inline-block; width:22px; height:4px; background:{COLOR_CANDIDATE_ARC}; margin-right:8px;"></span>
                <span><b>KD-Tree Candidates</b> (35m radius)</span>
            </div>
            <div style="display:flex; align-items:center; margin-bottom: 5px;">
                <span style="display:inline-block; width:22px; height:5px; background:{COLOR_GROUND_TRUTH}; margin-right:8px;"></span>
                <span><b>Ground-Truth Route</b> {f'({len(reference_route_arcs)} arcs)' if has_gt else '(Not Supplied)'}</span>
            </div>
            <div style="display:flex; align-items:center;">
                <span style="display:inline-block; width:22px; height:4px; border-top: 3px dashed {COLOR_PREDICTED_ROUTE}; margin-right:8px;"></span>
                <span><b>Predicted Route</b> {f'({len(predicted_route_arcs)} arcs)' if has_pred else '(Pending HMM Phase 4)'}</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(branca.element.Element(legend_html))

        # 9. Add Layer Control & Fit Bounds
        folium.LayerControl(position="topright", collapsed=False).add_to(m)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        return m

    def save_map(self, map_obj: folium.Map, output_path: str) -> str:
        """
        Saves the Folium map to an HTML file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        map_obj.save(output_path)
        return os.path.abspath(output_path)

    def generate_screenshot(
        self,
        html_path: str,
        screenshot_path: str,
        width: int = 1440,
        height: int = 900,
        wait_seconds: int = 3,
    ) -> Optional[str]:
        """
        Renders a high-resolution screenshot of the HTML map using Chrome/Edge headless CLI.
        """
        os.makedirs(os.path.dirname(os.path.abspath(screenshot_path)), exist_ok=True)
        abs_html = os.path.abspath(html_path)
        abs_png = os.path.abspath(screenshot_path)
        file_url = f"file:///{abs_html.replace(os.sep, '/')}"

        # Browser binary candidates
        chrome_bins = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]

        browser_bin = None
        for b in chrome_bins:
            if os.path.exists(b):
                browser_bin = b
                break

        if browser_bin is None:
            print(f"[Warning] No Chrome/Edge binary found for screenshot generation.")
            return None

        cmd = [
            browser_bin,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            f"--window-size={width},{height}",
            f"--screenshot={abs_png}",
            file_url,
        ]

        try:
            # Short sleep before screenshot to allow Leaflet tiles to load
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
            # Check if screenshot was created
            if os.path.exists(abs_png) and os.path.getsize(abs_png) > 1000:
                return abs_png
            else:
                print(f"[Notice] Screenshot command returned code {res.returncode}. Stderr: {res.stderr.decode('utf-8', errors='ignore')[:200]}")
                return None
        except Exception as e:
            print(f"[Warning] Failed to generate screenshot: {e}")
            return None


def visualize_trip(
    trip_dir: str,
    output_dir: str = "pathsnap/artifacts/maps",
    trip_id: Optional[str] = None,
    predicted_route_arcs: Optional[List[int]] = None,
    generate_screenshot: bool = True,
    spatial_search_radius_m: float = 35.0,
) -> Dict[str, Any]:
    """
    High-level convenience API to load trip data, build interactive map,
    save HTML, and optionally capture high-resolution screenshot.

    Parameters
    ----------
    trip_dir : str
        Directory containing .track, .nodes, .arcs, and optional .route.
    output_dir : str
        Target folder inside pathsnap/artifacts/maps/.
    trip_id : str, optional
        Trip identifier. Inferred from directory name if None.
    predicted_route_arcs : list of int, optional
        Optional predicted route arc IDs (for Phase 4 HMM/Viterbi evaluation).
    generate_screenshot : bool
        Whether to generate high-resolution PNG screenshot.
    spatial_search_radius_m : float
        Search radius in meters for candidate arc retrieval.

    Returns
    -------
    dict with:
        trip_id : str
        html_path : str
        screenshot_path : str or None
        num_observations : int
        num_candidates : int
        has_ground_truth : bool
    """
    if trip_id is None:
        trip_id = os.path.basename(os.path.normpath(trip_dir))

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    # 1. Preprocess trajectory
    gps_prep = GPSPreprocessor()
    trajectory = gps_prep.load_and_process(track_path, trip_id=trip_id)

    # 2. Preprocess network using trajectory projector
    net_prep = NetworkPreprocessor()
    network = net_prep.load_and_process(nodes_path, arcs_path, projector=trajectory.projector, trip_id=trip_id)

    # 3. Query spatial candidates
    spatial_idx = SampledArcKDTree(
        sampling_interval_m=25.0,
        search_radius_m=spatial_search_radius_m,
        fallback_k=15,
    ).build(network)
    candidates_per_obs = spatial_idx.query_trajectory(trajectory.coords_metric)

    # 4. Load ground-truth route if present
    reference_route_arcs = None
    has_gt = os.path.exists(route_path)
    if has_gt:
        raw_route = np.loadtxt(route_path, dtype=int)
        if raw_route.ndim == 0:
            raw_route = np.array([int(raw_route)])
        reference_route_arcs = list(raw_route)

    # 5. Build Map
    visualizer = MapVisualizer()
    folium_map = visualizer.create_trip_map(
        trajectory=trajectory,
        network=network,
        candidate_matches=candidates_per_obs,
        reference_route_arcs=reference_route_arcs,
        predicted_route_arcs=predicted_route_arcs,
        title=f"PathSnap Map Matching — Trip {trip_id}",
    )

    # 6. Save HTML Map
    html_filename = f"trip_{trip_id}_map.html"
    html_path = os.path.join(output_dir, html_filename)
    saved_html = visualizer.save_map(folium_map, html_path)

    # 7. Generate Screenshot
    png_path = None
    if generate_screenshot:
        png_filename = f"trip_{trip_id}_map_screenshot.png"
        target_png = os.path.join(output_dir, png_filename)
        png_path = visualizer.generate_screenshot(saved_html, target_png)

    total_cands = sum(len(c) for c in candidates_per_obs)

    return {
        "trip_id": trip_id,
        "html_path": saved_html,
        "screenshot_path": png_path,
        "num_observations": len(trajectory.df),
        "total_candidates": total_cands,
        "has_ground_truth": has_gt,
        "has_prediction": predicted_route_arcs is not None,
    }
