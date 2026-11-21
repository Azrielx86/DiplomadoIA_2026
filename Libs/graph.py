"""
Generic Graph implementation for Python 3.14+.
Supports C++/C# style generic typing: Graph[T]().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator

from .algorithms import AStarResult, a_star
from .edge import Edge
from .node import Node


class Graph[T]:
    """A generic graph data structure parameterized by node payload type T.

    Supports directed or undirected graphs with weighted edges (distances).

    Example usage:
        >>> g = Graph[int]()
        >>> node_a = g.add_node("A", 100)
        >>> node_b = g.add_node("B", 200)
        >>> g.add_edge("A", "B", distance=5.5)
        >>> g.get_edge_distance("A", "B")
        5.5
        >>> result = g.a_star("A", "B")
        >>> result.path_ids
        ['A', 'B']
    """

    def __init__(self, directed: bool = False) -> None:
        """Initialize a Graph.

        Args:
            directed: If True, edges are directed by default. If False, undirected.
        """
        self.directed: bool = directed
        self._nodes: dict[str, Node[T]] = {}

    # -------------------------------------------------------------------------
    # Node Management Methods
    # -------------------------------------------------------------------------

    def add_node(
        self,
        node_or_id: str | Node[T] | None = None,
        data: T | None = None,
        *,
        value: T | None = None,
    ) -> Node[T]:
        """Add a node to the graph or update an existing node's data.

        Args:
            node_or_id: Either an existing Node[T] instance, a string identifier,
                        or None to auto-generate a unique node ID.
            data: Payload of generic type T associated with the node.
            value: Synonym for data.

        Returns:
            The added or updated Node[T].
        """
        payload = data if data is not None else value

        if isinstance(node_or_id, Node):
            node = node_or_id
            if payload is not None:
                node.data = payload
        else:
            node_id = str(node_or_id) if node_or_id is not None else None
            if node_id is not None and node_id in self._nodes:
                existing = self._nodes[node_id]
                if payload is not None:
                    existing.data = payload
                return existing
            node = Node[T](id=node_id, data=payload)

        self._nodes[node.id] = node
        return node

    def get_node(self, id: str) -> Node[T] | None:
        """Retrieve a node by its string ID, or None if not found."""
        return self._nodes.get(str(id))

    def find_node(self, id: str) -> Node[T] | None:
        """Alias for get_node."""
        return self.get_node(id)

    def has_node(self, node_or_id: str | Node[T]) -> bool:
        """Check if a node exists in the graph."""
        node_id = node_or_id.id if isinstance(node_or_id, Node) else str(node_or_id)
        return node_id in self._nodes

    def remove_node(self, node_or_id: str | Node[T]) -> bool:
        """Remove a node and all its incoming and outgoing edges from the graph.

        Returns:
            True if the node was removed, False if it was not found.
        """
        node_id = node_or_id.id if isinstance(node_or_id, Node) else str(node_or_id)
        if node_id not in self._nodes:
            return False

        # Remove the node
        del self._nodes[node_id]

        # Remove all incoming edges pointing to this node from other nodes
        for node in self._nodes.values():
            node.remove_neighbor(node_id)

        return True

    def find_nodes(self, predicate: Callable[[Node[T]], bool]) -> list[Node[T]]:
        """Find all nodes satisfying a given predicate on Node[T]."""
        return [node for node in self._nodes.values() if predicate(node)]

    def find_nodes_by_data(self, predicate: Callable[[T | None], bool]) -> list[Node[T]]:
        """Find all nodes whose payload data satisfies a given predicate."""
        return [node for node in self._nodes.values() if predicate(node.data)]

    def find_nodes_by_value(self, value: T) -> list[Node[T]]:
        """Find all nodes whose data equals the specified value."""
        return [node for node in self._nodes.values() if node.data == value]

    # -------------------------------------------------------------------------
    # Edge & Distance Management Methods
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        source: str | Node[T],
        target: str | Node[T],
        distance: float = 1.0,
        bidirectional: bool | None = None,
    ) -> Edge[T]:
        """Add a weighted edge between two nodes with the given distance.

        If either node does not exist in the graph, it will be automatically created.

        Args:
            source: Source node or string ID.
            target: Target node or string ID.
            distance: Distance / weight between the nodes (default: 1.0).
            bidirectional: If True, adds edges both ways. If None, defaults to
                           True for undirected graphs and False for directed graphs.

        Returns:
            The created Edge[T] object.
        """
        src_id = source.id if isinstance(source, Node) else str(source)
        tgt_id = target.id if isinstance(target, Node) else str(target)

        # Auto-create nodes if not present
        if src_id not in self._nodes:
            src_node = source if isinstance(source, Node) else Node[T](src_id)
            self._nodes[src_id] = src_node
        else:
            src_node = self._nodes[src_id]

        if tgt_id not in self._nodes:
            tgt_node = target if isinstance(target, Node) else Node[T](tgt_id)
            self._nodes[tgt_id] = tgt_node
        else:
            tgt_node = self._nodes[tgt_id]

        dist = float(distance)
        src_node.add_neighbor(tgt_id, dist)

        is_both = (not self.directed) if bidirectional is None else bidirectional
        if is_both:
            tgt_node.add_neighbor(src_id, dist)

        return Edge(src_node, tgt_node, distance=dist, directed=not is_both)

    def remove_edge(
        self,
        source: str | Node[T],
        target: str | Node[T],
        bidirectional: bool | None = None,
    ) -> bool:
        """Remove the edge between source and target.

        Returns:
            True if an edge was removed, False otherwise.
        """
        src_id = source.id if isinstance(source, Node) else str(source)
        tgt_id = target.id if isinstance(target, Node) else str(target)

        src_node = self.get_node(src_id)
        if src_node is None:
            return False

        removed = src_node.remove_neighbor(tgt_id)

        is_both = (not self.directed) if bidirectional is None else bidirectional
        if is_both:
            tgt_node = self.get_node(tgt_id)
            if tgt_node is not None:
                removed_reverse = tgt_node.remove_neighbor(src_id)
                removed = removed or removed_reverse

        return removed

    def has_edge(self, source: str | Node[T], target: str | Node[T]) -> bool:
        """Check if an edge exists from source to target."""
        src_id = source.id if isinstance(source, Node) else str(source)
        tgt_id = target.id if isinstance(target, Node) else str(target)

        src_node = self.get_node(src_id)
        if src_node is None:
            return False
        return src_node.has_neighbor(tgt_id)

    def get_edge_distance(
        self,
        source: str | Node[T],
        target: str | Node[T],
    ) -> float | None:
        """Get the distance/weight of the edge between source and target.

        Returns:
            Distance as float, or None if no edge exists.
        """
        src_id = source.id if isinstance(source, Node) else str(source)
        tgt_id = target.id if isinstance(target, Node) else str(target)

        src_node = self.get_node(src_id)
        if src_node is None:
            return None
        return src_node.get_distance_to(tgt_id)

    def set_edge_distance(
        self,
        source: str | Node[T],
        target: str | Node[T],
        distance: float,
        bidirectional: bool | None = None,
    ) -> bool:
        """Update the distance between two existing connected nodes.

        Returns:
            True if distance was updated, False if edge does not exist.
        """
        if not self.has_edge(source, target):
            return False

        self.add_edge(source, target, distance=distance, bidirectional=bidirectional)
        return True

    # -------------------------------------------------------------------------
    # Inspection & Adjacency Methods
    # -------------------------------------------------------------------------

    def get_neighbors(self, node_or_id: str | Node[T]) -> dict[Node[T], float]:
        """Get neighboring nodes mapped to their edge distances.

        Returns:
            dict mapping Node[T] -> distance. Empty dict if node not found.
        """
        node_id = node_or_id.id if isinstance(node_or_id, Node) else str(node_or_id)
        node = self.get_node(node_id)
        if node is None:
            return {}

        result: dict[Node[T], float] = {}
        for neighbor_id, dist in node.neighbors.items():
            neighbor_node = self.get_node(neighbor_id)
            if neighbor_node is not None:
                result[neighbor_node] = dist
        return result

    def get_neighbor_nodes(self, node_or_id: str | Node[T]) -> list[Node[T]]:
        """Get a list of neighboring Node[T] objects."""
        return list(self.get_neighbors(node_or_id).keys())

    @property
    def nodes(self) -> dict[str, Node[T]]:
        """Dictionary of all nodes in the graph keyed by node ID."""
        return self._nodes.copy()

    def get_nodes(self) -> list[Node[T]]:
        """List of all Node[T] instances in the graph."""
        return list(self._nodes.values())

    def get_edges(self) -> list[Edge[T]]:
        """List of all Edge[T] instances in the graph."""
        edges: list[Edge[T]] = []
        seen: set[tuple[str, str]] = set()

        for src_node in self._nodes.values():
            for tgt_id, dist in src_node.neighbors.items():
                tgt_node = self._nodes.get(tgt_id)
                if tgt_node is None:
                    continue

                if self.directed:
                    edges.append(Edge(src_node, tgt_node, distance=dist, directed=True))
                else:
                    pair = tuple(sorted([src_node.id, tgt_node.id]))
                    if pair not in seen:
                        seen.add(pair)
                        edges.append(Edge(src_node, tgt_node, distance=dist, directed=False))

        return edges

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.get_edges())

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._nodes.clear()

    # -------------------------------------------------------------------------
    # Algorithm Methods
    # -------------------------------------------------------------------------

    def a_star(
        self,
        start: str | Node[T],
        goal: str | Node[T],
        heuristic: (
            Callable[[Node[T], Node[T]], float]
            | Callable[[Node[T]], float]
            | None
        ) = None,
    ) -> AStarResult[T] | None:
        """Find shortest path between start and goal using the A* algorithm.

        Args:
            start: Start node or string ID.
            goal: Target node or string ID.
            heuristic: Optional heuristic function.

        Returns:
            AStarResult[T] if path found, or None if no path exists.
        """
        return a_star(self, start, goal, heuristic=heuristic)

    # -------------------------------------------------------------------------
    # Visualization Methods (Matplotlib)
    # -------------------------------------------------------------------------

    def _resolve_positions(
        self,
        positions: (
            dict[str, tuple[float, float]]
            | Callable[[Node[T]], tuple[float, float]]
            | None
        ) = None,
    ) -> dict[str, tuple[float, float]]:
        """Resolves 2D (x, y) coordinates for all nodes in the graph."""
        if isinstance(positions, dict):
            return {str(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
        if callable(positions):
            return {node_id: positions(node) for node_id, node in self._nodes.items()}

        # Attempt to auto-extract from node.data (Point, tuple, dict, etc.)
        extracted_pos: dict[str, tuple[float, float]] = {}
        can_extract_all = bool(self._nodes)
        for node_id, node in self._nodes.items():
            data = node.data
            if data is None:
                can_extract_all = False
                break
            if hasattr(data, "x") and hasattr(data, "y"):
                try:
                    extracted_pos[node_id] = (float(data.x), float(data.y))
                    continue
                except (ValueError, TypeError):
                    pass
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                try:
                    extracted_pos[node_id] = (float(data[0]), float(data[1]))
                    continue
                except (ValueError, TypeError):
                    pass
            if isinstance(data, dict):
                if "x" in data and "y" in data:
                    try:
                        extracted_pos[node_id] = (float(data["x"]), float(data["y"]))
                        continue
                    except (ValueError, TypeError):
                        pass
                elif "lon" in data and "lat" in data:
                    try:
                        extracted_pos[node_id] = (float(data["lon"]), float(data["lat"]))
                        continue
                    except (ValueError, TypeError):
                        pass
            can_extract_all = False
            break

        if can_extract_all and len(extracted_pos) == len(self._nodes):
            return extracted_pos

        # Default fallback: circular layout
        import math

        n = len(self._nodes)
        pos: dict[str, tuple[float, float]] = {}
        for idx, node_id in enumerate(self._nodes.keys()):
            angle = (2 * math.pi * idx / max(n, 1)) + (math.pi / 2)
            pos[node_id] = (math.cos(angle), math.sin(angle))
        return pos

    def plot(
        self,
        positions: (
            dict[str, tuple[float, float]]
            | Callable[[Node[T]], tuple[float, float]]
            | None
        ) = None,
        highlight_path: (
            list[Node[T]]
            | list[str]
            | AStarResult[T]
            | None
        ) = None,
        show_distances: bool = True,
        title: str = "Graph Visualization",
        node_color: str = "#87CEEB",
        highlight_color: str = "#FF5722",
        edge_color: str = "#888888",
        node_size: float = 650.0,
        font_size: int = 10,
        figsize: tuple[float, float] = (8.0, 6.0),
        ax: Any | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> Any:
        """Plot the graph using matplotlib.

        Args:
            positions: Optional node coordinates. Can be a dict mapping node_id -> (x, y),
                       or a callable `f(node) -> (x, y)`. If None, automatically detects
                       coordinates from node data, or falls back to a circular layout.
            highlight_path: Optional path to highlight (e.g. from A* search result or list of nodes/IDs).
            show_distances: Whether to display distance/weight annotations on edges.
            title: Title for the matplotlib figure.
            node_color: Background color for nodes (hex or color name).
            highlight_color: Color used for nodes and edges in the highlighted path.
            edge_color: Color for standard graph edges.
            node_size: Size of the node markers.
            font_size: Base font size for node labels.
            figsize: Figure size in inches (width, height) if a new figure is created.
            ax: Optional existing matplotlib Axes object. If None, creates a new figure.
            show: If True, calls `plt.show()`.
            save_path: Optional file path to save the generated plot image.

        Returns:
            The matplotlib Axes object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. Install it using: pip install matplotlib"
            ) from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        pos = self._resolve_positions(positions)

        # Parse highlight path
        highlight_edges: set[tuple[str, str]] = set()
        highlight_nodes: set[str] = set()

        if highlight_path is not None:
            if hasattr(highlight_path, "path_ids"):
                path_ids = list(highlight_path.path_ids)
            elif isinstance(highlight_path, (list, tuple)):
                path_ids = [n.id if hasattr(n, "id") else str(n) for n in highlight_path]
            else:
                path_ids = []

            highlight_nodes = set(path_ids)
            for i in range(len(path_ids) - 1):
                u, v = path_ids[i], path_ids[i + 1]
                highlight_edges.add((u, v))
                if not self.directed:
                    highlight_edges.add((v, u))

        # 1. Draw Edges
        for edge in self.get_edges():
            u_id = edge.source.id
            v_id = edge.target.id
            if u_id not in pos or v_id not in pos:
                continue

            x1, y1 = pos[u_id]
            x2, y2 = pos[v_id]

            is_hl = (u_id, v_id) in highlight_edges or (
                not self.directed and (v_id, u_id) in highlight_edges
            )
            col = highlight_color if is_hl else edge_color
            lw = 2.5 if is_hl else 1.2
            alpha = 1.0 if is_hl else 0.7

            if self.directed:
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=lw,
                        alpha=alpha,
                        shrinkA=16,
                        shrinkB=16,
                        mutation_scale=14,
                    ),
                    zorder=2,
                )
            else:
                ax.plot([x1, x2], [y1, y2], color=col, linewidth=lw, alpha=alpha, zorder=2)

            # Edge distance label
            if show_distances:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dist_val = edge.distance
                dist_str = (
                    f"{dist_val:.1f}"
                    if isinstance(dist_val, float) and not dist_val.is_integer()
                    else f"{int(dist_val)}"
                )
                ax.text(
                    mid_x,
                    mid_y,
                    dist_str,
                    fontsize=max(font_size - 2, 7),
                    color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="#dddddd",
                        alpha=0.85,
                        linewidth=0.5,
                    ),
                    ha="center",
                    va="center",
                    zorder=3,
                )

        # 2. Draw Nodes
        for node_id, node in self._nodes.items():
            if node_id not in pos:
                continue
            x, y = pos[node_id]
            is_hl = node_id in highlight_nodes
            col = highlight_color if is_hl else node_color
            edge_col = "#990000" if is_hl else "#333333"

            ax.scatter(
                [x],
                [y],
                s=node_size,
                color=col,
                edgecolors=edge_col,
                linewidths=2.0 if is_hl else 1.5,
                zorder=4,
            )

            # Node label (ID)
            ax.text(
                x,
                y,
                str(node.id),
                fontsize=font_size,
                fontweight="bold",
                color="white" if is_hl else "#111111",
                ha="center",
                va="center",
                zorder=5,
            )

        # Set title and clean layout
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=12)
        ax.axis("off")

        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            dx = (max(xs) - min(xs)) * 0.15 if len(xs) > 1 and max(xs) != min(xs) else 0.5
            dy = (max(ys) - min(ys)) * 0.15 if len(ys) > 1 and max(ys) != min(ys) else 0.5
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        if show:
            plt.show()

        return ax

    # -------------------------------------------------------------------------
    # Python Container Protocols
    # -------------------------------------------------------------------------

    def __contains__(self, node_or_id: str | Node[T]) -> bool:
        return self.has_node(node_or_id)

    def __getitem__(self, node_id: str) -> Node[T]:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node {node_id!r} not found in graph.")
        return node

    def __setitem__(self, node_id: str, data: T) -> None:
        self.add_node(id_or_node=node_id, data=data)

    def __delitem__(self, node_id: str) -> None:
        if not self.remove_node(node_id):
            raise KeyError(f"Node {node_id!r} not found in graph.")

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node[T]]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        kind = "DirectedGraph" if self.directed else "Graph"
        return f"{kind}[T](nodes={self.num_nodes}, edges={self.num_edges})"

    def __str__(self) -> str:
        return f"{self.__repr__()}"
