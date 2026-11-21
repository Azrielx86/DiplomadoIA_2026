"""
Libs - Generic Graph Library for Python 3.14+
=============================================

A simple and powerful library to create graphs with generic types, nodes identified
by string IDs, weighted distances between nodes, and pathfinding algorithms including A*.

Example:
    from Libs import Graph, Node, a_star

    # Create a generic graph with tuple coordinates (x, y)
    g = Graph[tuple[float, float]]()

    # Add nodes with string IDs and data
    g.add_node("A", (0.0, 0.0))
    g.add_node("B", (3.0, 4.0))

    # Add distance between nodes
    g.add_edge("A", "B", distance=5.0)

    # Find path using A*
    result = g.a_star("A", "B")
    print(result.path_ids)      # ['A', 'B']
    print(result.total_distance) # 5.0
"""

from .algorithms import AStarResult, a_star
from .edge import Edge
from .graph import Graph
from .node import Node

__all__ = [
    "Node",
    "Graph",
    "Edge",
    "a_star",
    "AStarResult",
]
