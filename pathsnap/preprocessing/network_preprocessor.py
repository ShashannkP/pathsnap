"""
pathsnap.preprocessing.network_preprocessor
===========================================
Loads, validates, projects, and constructs the directed road-network graph.

Strictly preserves original 0-based node_id and arc_id values while computing
spatial segment lengths, bearings, adjacency lists, and network components.
"""

import os
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional
from ..geospatial.projection import LocalMetricProjector, compute_bearing


class RoadNetwork:
    """
    Holds projected road-network geometry, directed graph topology, and cached spatial metrics.
    """

    def __init__(
        self,
        trip_id: str,
        nodes_lonlat: np.ndarray,
        nodes_metric: np.ndarray,
        arcs: np.ndarray,
        arc_lengths: np.ndarray,
        arc_bearings: np.ndarray,
        graph: nx.DiGraph,
        metadata: dict,
    ):
        self.trip_id = trip_id
        self.nodes_lonlat = nodes_lonlat    # (N, 2) [lon, lat]
        self.nodes_metric = nodes_metric    # (N, 2) [x_m, y_m]
        self.arcs = arcs                    # (M, 2) [u, v]
        self.arc_lengths = arc_lengths      # (M,) in meters
        self.arc_bearings = arc_bearings    # (M,) in radians [0, 2pi)
        self.graph = graph                  # networkx.DiGraph
        self.metadata = metadata

        # Cached endpoints for fast vectorized distance calculations
        self.u_coords = self.nodes_metric[self.arcs[:, 0]]
        self.v_coords = self.nodes_metric[self.arcs[:, 1]]

        # Cached adjacency lookups
        self.num_nodes = len(nodes_lonlat)
        self.num_arcs = len(arcs)

        # Map node to outgoing and incoming arc IDs
        self.node_outgoing_arcs: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        self.node_incoming_arcs: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        for arc_id in range(self.num_arcs):
            u, v = self.arcs[arc_id]
            self.node_outgoing_arcs[u].append(arc_id)
            self.node_incoming_arcs[v].append(arc_id)


class NetworkPreprocessor:
    """
    Loads, cleans, validates, and builds spatial road networks from .nodes and .arcs files.
    """

    def __init__(self, min_arc_length_eps: float = 1e-3):
        self.min_arc_length_eps = min_arc_length_eps

    def load_and_process(
        self,
        nodes_path: str,
        arcs_path: str,
        projector: LocalMetricProjector,
        trip_id: Optional[str] = None,
    ) -> RoadNetwork:
        """
        Loads and validates road network files.

        Parameters
        ----------
        nodes_path : str
            Path to .nodes file.
        arcs_path : str
            Path to .arcs file.
        projector : LocalMetricProjector
            Projector initialized to the trip's coordinate system.
        trip_id : str, optional
            Trip identifier.

        Returns
        -------
        RoadNetwork
        """
        if trip_id is None:
            trip_id = os.path.splitext(os.path.basename(nodes_path))[0]

        # Load nodes (2 columns: lon, lat)
        nodes_raw = np.loadtxt(nodes_path, dtype=float)
        if nodes_raw.ndim == 1:
            nodes_raw = nodes_raw.reshape(1, 2)
        num_nodes = len(nodes_raw)

        # Load arcs (2 columns: from_node_id, to_node_id)
        arcs_raw = np.loadtxt(arcs_path, dtype=int)
        if arcs_raw.ndim == 1:
            arcs_raw = arcs_raw.reshape(1, 2)
        num_arcs = len(arcs_raw)

        # Integrity Validation: Node boundary references
        u_max = arcs_raw[:, 0].max() if num_arcs > 0 else 0
        v_max = arcs_raw[:, 1].max() if num_arcs > 0 else 0
        if u_max >= num_nodes or v_max >= num_nodes or arcs_raw.min() < 0:
            raise ValueError(
                f"Trip {trip_id} has invalid arc references: max u={u_max}, max v={v_max}, total nodes={num_nodes}"
            )

        # Project node coordinates to metric plane
        nodes_metric = projector.project(nodes_raw[:, 0], nodes_raw[:, 1])

        # Compute metric geometry for all arcs
        u_coords = nodes_metric[arcs_raw[:, 0]]
        v_coords = nodes_metric[arcs_raw[:, 1]]
        dxy = v_coords - u_coords
        arc_lengths = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)

        # Compute road bearings
        arc_bearings = np.zeros(num_arcs, dtype=float)
        for i in range(num_arcs):
            arc_bearings[i] = compute_bearing(dxy[i, 0], dxy[i, 1])

        # Build directed networkx graph
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_nodes))
        for arc_id in range(num_arcs):
            u, v = int(arcs_raw[arc_id, 0]), int(arcs_raw[arc_id, 1])
            length = float(arc_lengths[arc_id])
            graph.add_edge(u, v, arc_id=arc_id, weight=max(length, self.min_arc_length_eps))

        # Check graph metrics
        isolated_nodes = list(nx.isolates(graph))
        num_strongly_connected = nx.number_strongly_connected_components(graph)
        num_weakly_connected = nx.number_weakly_connected_components(graph)
        duplicate_arcs = num_arcs - graph.number_of_edges()

        metadata = {
            "trip_id": trip_id,
            "num_nodes": num_nodes,
            "num_arcs": num_arcs,
            "unique_directed_edges": graph.number_of_edges(),
            "duplicate_arcs_count": duplicate_arcs,
            "isolated_nodes_count": len(isolated_nodes),
            "num_strongly_connected_components": num_strongly_connected,
            "num_weakly_connected_components": num_weakly_connected,
            "mean_arc_length_m": float(arc_lengths.mean()) if num_arcs > 0 else 0.0,
            "median_arc_length_m": float(np.median(arc_lengths)) if num_arcs > 0 else 0.0,
            "min_arc_length_m": float(arc_lengths.min()) if num_arcs > 0 else 0.0,
            "max_arc_length_m": float(arc_lengths.max()) if num_arcs > 0 else 0.0,
        }

        return RoadNetwork(
            trip_id=trip_id,
            nodes_lonlat=nodes_raw,
            nodes_metric=nodes_metric,
            arcs=arcs_raw,
            arc_lengths=arc_lengths,
            arc_bearings=arc_bearings,
            graph=graph,
            metadata=metadata,
        )
