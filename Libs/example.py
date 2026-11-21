"""
Example usage of the generic graph library in Python 3.14.
Demonstrates Node[T](), Graph[T](), distance management, node queries, and A* algorithm.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure Libs can be imported regardless of current working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Libs import Graph, Node, a_star


# 1. Define a generic data payload (e.g., City with 2D coordinates)
@dataclass
class CityInfo:
    name: str
    x: float
    y: float
    population: int


def euclidean_distance(city1: CityInfo, city2: CityInfo) -> float:
    """Straight-line (Euclidean) distance between two cities."""
    return math.hypot(city1.x - city2.x, city1.y - city2.y)


def main() -> None:
    print("=" * 60)
    print("Generic Graph Library Demo (Python 3.14 - Node[T], Graph[T])")
    print("=" * 60)

    # 2. Instantiate a generic Graph typed to CityInfo
    # In Python 3.14, Graph[T]() uses PEP 695 generics
    graph: Graph[CityInfo] = Graph[CityInfo](directed=False)

    # 3. Add nodes identified by string IDs with generic CityInfo payloads
    graph.add_node("MEX", CityInfo(name="Mexico City", x=19.4, y=-99.1, population=9_200_000))
    graph.add_node("GDL", CityInfo(name="Guadalajara", x=20.6, y=-103.3, population=1_500_000))
    graph.add_node("MTY", CityInfo(name="Monterrey", x=25.6, y=-100.3, population=1_100_000))
    graph.add_node("QRO", CityInfo(name="Queretaro", x=20.5, y=-100.3, population=1_050_000))
    graph.add_node("SLP", CityInfo(name="San Luis Potosi", x=22.1, y=-100.9, population=900_000))
    graph.add_node("LEO", CityInfo(name="Leon", x=21.1, y=-101.6, population=1_700_000))

    # 4. Add road distances (km) between nodes
    graph.add_edge("MEX", "QRO", distance=218.0)
    graph.add_edge("QRO", "LEO", distance=165.0)
    graph.add_edge("LEO", "GDL", distance=220.0)
    graph.add_edge("QRO", "SLP", distance=203.0)
    graph.add_edge("SLP", "MTY", distance=514.0)
    graph.add_edge("SLP", "GDL", distance=340.0)
    graph.add_edge("GDL", "MTY", distance=780.0)

    print(f"\n[+] Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges.")

    # 5. Finding nodes
    print("\n--- Finding Nodes ---")
    mex_node = graph.get_node("MEX")
    print(f"Find by ID 'MEX': {mex_node.data.name} (Pop: {mex_node.data.population:,})")

    # Find nodes by predicate on data (e.g., population > 1,400,000)
    large_cities = graph.find_nodes_by_data(
        lambda data: data is not None and data.population > 1_400_000
    )
    print(f"Cities with population > 1.4M: {[n.id for n in large_cities]}")

    # Inspect neighbors of Queretaro
    print("\n--- Neighbor Inspection ---")
    neighbors = graph.get_neighbors("QRO")
    for neighbor, dist in neighbors.items():
        print(f"  QRO <-> {neighbor.id} ({neighbor.data.name}): {dist} km")

    # 6. Run A* Pathfinding from Mexico City (MEX) to Monterrey (MTY)
    print("\n--- A* Pathfinding (MEX -> MTY) ---")

    # Admissible Heuristic function: Euclidean distance converted to approx km (1 deg ~ 111 km)
    def a_star_heuristic(current: Node[CityInfo], goal: Node[CityInfo]) -> float:
        if current.data is None or goal.data is None:
            return 0.0
        degree_dist = euclidean_distance(current.data, goal.data)
        return degree_dist * 105.0  # Under-estimate in km to maintain admissibility

    # Execute A* search
    result = graph.a_star("MEX", "MTY", heuristic=a_star_heuristic)

    if result:
        print(f"Shortest path found: {' -> '.join(result.path_ids)}")
        print(f"Total travel distance: {result.total_distance:.1f} km")
        print(f"Explored nodes count: {result.visited_count}")

        # Tuple unpacking test:
        path, distance = result
        print("\nStep-by-step route:")
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            seg_dist = graph.get_edge_distance(n1.id, n2.id)
            print(f"  {n1.id} ({n1.data.name}) -> {n2.id} ({n2.data.name}): {seg_dist:.1f} km")

        # 7. Plot the graph and highlight the A* shortest path
        print("\n--- Visualizing Graph with Matplotlib ---")
        plot_path = Path(__file__).resolve().parent / "graph_demo.png"
        graph.plot(
            positions=lambda n: (n.data.y, n.data.x) if n.data else (0.0, 0.0),
            highlight_path=result,
            title="Road Network - A* Shortest Path (MEX -> MTY)",
            save_path=str(plot_path),
            show=False,
        )
        print(f"Saved graph visualization to: {plot_path}")
    else:
        print("No route found!")


if __name__ == "__main__":
    main()
