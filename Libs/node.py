"""
Generic Node implementation for Python 3.14+.
Supports C++/C# style generic typing: Node[T]().
"""

from __future__ import annotations

import itertools
from typing import Any


class Node[T]:
    """A generic node in a graph identified by a unique string ID.

    Attributes:
        id: Unique string identifier for the node.
        data: Generic data/payload of type T associated with the node.
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        id: str | None = None,
        data: T | None = None,
        *,
        value: T | None = None,
    ) -> None:
        """Initialize a Node.

        Args:
            id: Unique identifier for the node. If None, auto-generates a string ID.
            data: Generic payload of type T stored in this node.
            value: Alias for data.
        """
        if id is None:
            self.id: str = f"node_{next(self._id_counter)}"
        else:
            self.id = str(id)

        # Allow passing payload via either 'data' or 'value'
        self.data: T | None = data if data is not None else value
        # Mapping neighbor_id -> distance (float)
        self._neighbors: dict[str, float] = {}

    @property
    def value(self) -> T | None:
        """Alias for data property."""
        return self.data

    @value.setter
    def value(self, new_val: T | None) -> None:
        self.data = new_val

    @property
    def neighbors(self) -> dict[str, float]:
        """Dictionary of neighbor IDs mapped to their edge distance."""
        return self._neighbors.copy()

    @property
    def degree(self) -> int:
        """Number of outgoing/connected neighbors."""
        return len(self._neighbors)

    def add_neighbor(self, neighbor_id: str, distance: float = 1.0) -> None:
        """Add or update a connection to a neighboring node by ID with distance."""
        self._neighbors[str(neighbor_id)] = float(distance)

    def remove_neighbor(self, neighbor_id: str) -> bool:
        """Remove a connection to a neighbor. Returns True if removed, False if not present."""
        return self._neighbors.pop(str(neighbor_id), None) is not None

    def has_neighbor(self, neighbor_id: str) -> bool:
        """Check if this node has a connection to the specified neighbor ID."""
        return str(neighbor_id) in self._neighbors

    def get_distance_to(self, neighbor_id: str) -> float | None:
        """Get distance to the specified neighbor, or None if not connected."""
        return self._neighbors.get(str(neighbor_id))

    def clear_neighbors(self) -> None:
        """Remove all neighbor connections from this node."""
        self._neighbors.clear()

    def __repr__(self) -> str:
        data_repr = f", data={self.data!r}" if self.data is not None else ""
        return f"Node[T](id={self.id!r}{data_repr})"

    def __str__(self) -> str:
        return f"Node({self.id})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return False

    def __hash__(self) -> int:
        return hash(self.id)

    def __lt__(self, other: Any) -> bool:
        """Allows nodes to be deterministically ordered/compared by ID (useful in heapq)."""
        if isinstance(other, Node):
            return self.id < other.id
        return NotImplemented
