"""
Generic Edge implementation for Python 3.14+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import Node


class Edge[T]:
    """A weighted directed/undirected edge between two nodes in a graph.

    Attributes:
        source: Starting Node[T] of the edge.
        target: Destination Node[T] of the edge.
        distance: Distance or weight of the edge (float).
        directed: Whether this edge is directed.
    """

    def __init__(
        self,
        source: Node[T],
        target: Node[T],
        distance: float = 1.0,
        directed: bool = False,
    ) -> None:
        self.source = source
        self.target = target
        self.distance = float(distance)
        self.directed = directed

    @property
    def weight(self) -> float:
        """Alias for distance."""
        return self.distance

    @weight.setter
    def weight(self, value: float) -> None:
        self.distance = float(value)

    def __repr__(self) -> str:
        arrow = "->" if self.directed else "<->"
        return f"Edge({self.source.id} {arrow} {self.target.id}, distance={self.distance})"

    def __str__(self) -> str:
        arrow = "->" if self.directed else "<->"
        return f"({self.source.id} {arrow} {self.target.id}: {self.distance})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Edge):
            return False
        if self.directed != other.directed:
            return False
        if abs(self.distance - other.distance) > 1e-9:
            return False

        if self.directed:
            return self.source.id == other.source.id and self.target.id == other.target.id
        return (
            (self.source.id == other.source.id and self.target.id == other.target.id)
            or (self.source.id == other.target.id and self.target.id == other.source.id)
        )

    def __hash__(self) -> int:
        if self.directed:
            return hash((self.source.id, self.target.id, self.directed))
        return hash((frozenset({self.source.id, self.target.id}), self.directed))
