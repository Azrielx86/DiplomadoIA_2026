"""
Pathfinding and graph algorithms for generic graphs, including A* (A-Star).
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from .graph import Graph
    from .node import Node


class AStarResult[T]:
    """Result returned by the A* search algorithm.

    Attributes:
        path: List of Node[T] from start to goal in traversal order.
        total_distance: Total distance/weight of the shortest path found.
        visited_count: Number of nodes expanded during the search.
        found: True if a path was successfully found.
    """

    def __init__(
        self,
        path: list[Node[T]],
        total_distance: float,
        visited_count: int = 0,
        found: bool = True,
    ) -> None:
        self.path: list[Node[T]] = path
        self.total_distance: float = total_distance
        self.visited_count: int = visited_count
        self.found: bool = found

    @property
    def cost(self) -> float:
        """Alias for total_distance."""
        return self.total_distance

    @property
    def path_ids(self) -> list[str]:
        """List of node string IDs along the path."""
        return [node.id for node in self.path]

    def __bool__(self) -> bool:
        """Evaluates to True if a valid path was found."""
        return self.found and len(self.path) > 0

    def __len__(self) -> int:
        """Number of nodes in the path."""
        return len(self.path)

    def __getitem__(self, index: int) -> Node[T]:
        """Index into the path nodes."""
        return self.path[index]

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking: path, distance = a_star(...)"""
        yield self.path
        yield self.total_distance

    def __repr__(self) -> str:
        if not self.found:
            return "AStarResult(found=False, path=[], total_distance=inf)"
        ids_str = " -> ".join(self.path_ids)
        return (
            f"AStarResult(path=[{ids_str}], "
            f"total_distance={self.total_distance:.4f}, "
            f"visited={self.visited_count})"
        )


def _call_heuristic[T](
    heuristic: Callable[..., float] | None,
    current: Node[T],
    goal: Node[T],
) -> float:
    """Evaluates the heuristic function.

    Supports:
        - heuristic(current_node, goal_node) -> float
        - heuristic(current_node) -> float
        - None (returns 0.0, equivalent to Dijkstra's algorithm)
    """
    if heuristic is None:
        return 0.0

    try:
        # Try two-argument form: (current_node, goal_node)
        return float(heuristic(current, goal))
    except TypeError:
        try:
            # Try single-argument form: (current_node)
            return float(heuristic(current))
        except Exception as exc:
            raise TypeError(
                f"Heuristic function {heuristic} must accept either (current, goal) or (current)."
            ) from exc


def a_star[T](
    graph: Graph[T],
    start: str | Node[T],
    goal: str | Node[T],
    heuristic: (
        Callable[[Node[T], Node[T]], float]
        | Callable[[Node[T]], float]
        | None
    ) = None,
) -> AStarResult[T] | None:
    """Finds the shortest path between start and goal using the A* algorithm.

    Args:
        graph: The Graph[T] instance to search.
        start: Starting node or string ID.
        goal: Target destination node or string ID.
        heuristic: Optional heuristic function estimating distance to goal:
                   either h(current_node, goal_node) -> float or h(current_node) -> float.
                   If None, defaults to 0.0 (Dijkstra's shortest path).

    Returns:
        AStarResult[T] if a path is found, or None if no path exists.

    Raises:
        KeyError: If start or goal node does not exist in the graph.
    """
    start_id = start.id if hasattr(start, "id") else str(start)
    goal_id = goal.id if hasattr(goal, "id") else str(goal)

    start_node = graph.get_node(start_id)
    if start_node is None:
        raise KeyError(f"Start node {start_id!r} does not exist in the graph.")

    goal_node = graph.get_node(goal_id)
    if goal_node is None:
        raise KeyError(f"Goal node {goal_id!r} does not exist in the graph.")

    # Trivial case: start is goal
    if start_id == goal_id:
        return AStarResult(
            path=[start_node],
            total_distance=0.0,
            visited_count=1,
            found=True,
        )

    # Priority queue stores tuples: (f_score, counter, node_id)
    counter = itertools.count()
    open_set: list[tuple[float, int, str]] = []

    g_score: dict[str, float] = {start_id: 0.0}
    initial_h = _call_heuristic(heuristic, start_node, goal_node)
    f_score: dict[str, float] = {start_id: initial_h}

    heapq.heappush(open_set, (initial_h, next(counter), start_id))

    came_from: dict[str, str] = {}
    visited: set[str] = set()

    while open_set:
        current_f, _, current_id = heapq.heappop(open_set)

        if current_id in visited:
            continue
        visited.add(current_id)

        # Check if reached goal
        if current_id == goal_id:
            # Reconstruct path from came_from
            path_ids: list[str] = []
            curr: str | None = current_id
            while curr is not None:
                path_ids.append(curr)
                curr = came_from.get(curr)
            path_ids.reverse()

            path_nodes: list[Node[T]] = []
            for nid in path_ids:
                node = graph.get_node(nid)
                if node is not None:
                    path_nodes.append(node)

            return AStarResult(
                path=path_nodes,
                total_distance=g_score[goal_id],
                visited_count=len(visited),
                found=True,
            )

        current_node = graph.get_node(current_id)
        if current_node is None:
            continue

        current_g = g_score[current_id]

        for neighbor_id, distance in current_node.neighbors.items():
            if neighbor_id in visited:
                continue

            tentative_g = current_g + distance
            if tentative_g < g_score.get(neighbor_id, float("inf")):
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g

                neighbor_node = graph.get_node(neighbor_id)
                if neighbor_node is None:
                    continue

                h_val = _call_heuristic(heuristic, neighbor_node, goal_node)
                est_f = tentative_g + h_val
                f_score[neighbor_id] = est_f
                heapq.heappush(open_set, (est_f, next(counter), neighbor_id))

    # No path found
    return None
