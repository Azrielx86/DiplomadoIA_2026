# Generic Graph Library for Python 3.14+

A clean, modular Python 3.14+ library for working with graphs using modern generic types (PEP 695). It provides C++/C# style generic typing (`Node[T]()`, `Graph[T]()`), string-identified nodes, weighted distance edges, comprehensive querying and data interaction methods, and the **A\*** pathfinding algorithm.

---

## 📌 Comparison with C++/C# Notation

In C++/C#, generic classes use angle brackets:
```cpp
// C++ / C#
Node<T> node = new Node<T>("A", data);
Graph<T> graph = new Graph<T>();
```

In Python 3.14 (PEP 695 type parameters), the exact equivalent syntax uses square brackets:
```python
# Python 3.14
node = Node[T]("A", data)
graph = Graph[T]()
```

Both explicit parameterized form (`Node[T](...)`, `Graph[T]()`) and inferred form (`Node(...)`, `Graph()`) are fully supported.

---

## 🚀 Features

- **Generic Typing (`Node[T]`, `Graph[T]`)**: Associate any arbitrary data payload (numbers, strings, tuples, dictionaries, dataclasses) with nodes while maintaining strong type annotations.
- **String Node Identifiers**: Every node is uniquely identified by a string ID.
- **Weighted Edges / Distances**: Connect nodes with arbitrary distances (undirected or directed).
- **Rich Querying & Data Interaction**:
  - `get_node(id)` / `find_node(id)` / `graph[id]`
  - `find_nodes_by_data(predicate)`
  - `find_nodes_by_value(value)`
  - `find_nodes(predicate)`
  - `get_neighbors(id)` / `get_edge_distance(src, tgt)`
  - Update node values and edge distances on the fly.
- **A\* Pathfinding Algorithm**:
  - Fast priority-queue implementation (`heapq`).
  - Supports custom heuristic functions: `h(current, goal) -> float` or `h(current) -> float`.
  - Defaults to Dijkstra's algorithm (`h = 0.0`) when no heuristic is provided.
  - Returns structured `AStarResult[T]` that supports tuple unpacking (`path, distance = result`).

---

## 📂 Project Structure

```
Libs/
├── __init__.py      # Package exports (Node, Graph, Edge, a_star, AStarResult)
├── node.py          # Generic Node[T] implementation
├── edge.py          # Generic Edge[T] implementation
├── graph.py         # Generic Graph[T] implementation
├── algorithms.py    # A* algorithm and AStarResult[T]
├── example.py       # Educational runnable example script
├── test_graph.py    # Complete test suite (20 unit tests)
└── README.md        # Documentation and API reference
```

---

## 📖 Quick Start

### 1. Basic Graph Creation and Edges

```python
from Libs import Graph, Node

# Create an undirected graph storing integers
graph = Graph[int]()

# Add nodes with string IDs and integer payloads
graph.add_node("A", data=10)
graph.add_node("B", data=20)
graph.add_node("C", data=30)

# Add edges with distances
graph.add_edge("A", "B", distance=2.5)
graph.add_edge("B", "C", distance=4.0)
graph.add_edge("A", "C", distance=8.0)

print(graph.get_edge_distance("A", "B"))  # 2.5
```

### 2. Interacting with and Finding Nodes

```python
# Access nodes by string ID
node_a = graph.get_node("A")  # or graph["A"]
print(node_a.id, node_a.data)  # 'A', 10

# Update node data
node_a.data = 99
graph["B"].data = 150

# Check presence
if "A" in graph:
    print("Node A exists")

# Find nodes using predicates
high_value_nodes = graph.find_nodes_by_data(lambda data: data is not None and data > 50)
print([n.id for n in high_value_nodes])  # ['A', 'B']

# Inspect neighbors and distances
for neighbor_node, distance in graph.get_neighbors("A").items():
    print(f"Neighbor {neighbor_node.id} is {distance} units away")
```

### 3. A* Pathfinding

```python
from Libs import Graph, a_star

graph = Graph[str]()
graph.add_edge("Start", "Waypoint", distance=3.0)
graph.add_edge("Waypoint", "Goal", distance=2.0)
graph.add_edge("Start", "Goal", distance=10.0)

# Run A* search
result = graph.a_star("Start", "Goal")

if result:
    print("Path found:", result.path_ids)          # ['Start', 'Waypoint', 'Goal']
    print("Total distance:", result.total_distance) # 5.0

    # Or unpack directly as a tuple:
    path, distance = result
```

### 4. A* with Custom Heuristics (e.g., 2D Grid / Euclidean)

```python
import math
from dataclasses import dataclass
from Libs import Graph, Node

@dataclass
class Point:
    x: float
    y: float

graph = Graph[Point]()
graph.add_node("Origin", Point(0.0, 0.0))
graph.add_node("Mid", Point(2.0, 3.0))
graph.add_node("Dest", Point(5.0, 5.0))

graph.add_edge("Origin", "Mid", distance=3.6)
graph.add_edge("Mid", "Dest", distance=3.6)
graph.add_edge("Origin", "Dest", distance=10.0)

# Euclidean heuristic function
def euclidean_heuristic(curr: Node[Point], goal: Node[Point]) -> float:
    return math.hypot(curr.data.x - goal.data.x, curr.data.y - goal.data.y)

result = graph.a_star("Origin", "Dest", heuristic=euclidean_heuristic)
print(result.path_ids)       # ['Origin', 'Mid', 'Dest']
print(result.total_distance) # 7.2
```

### 5. Plotting the Graph with Matplotlib

```python
# Simple plot (uses circular layout automatically if no coordinates provided)
graph.plot(title="My Graph")

# Plot with custom positions and highlight an A* shortest path
result = graph.a_star("Origin", "Dest", heuristic=euclidean_heuristic)

graph.plot(
    positions={"Origin": (0, 0), "Mid": (2, 3), "Dest": (5, 5)},
    highlight_path=result,
    title="Shortest Path Visualization",
    save_path="shortest_path.png",  # Save to file
    show=True,
)
```

---

## 🧪 Running Tests & Examples

Run the unit tests:
```bash
python3 -m unittest Libs.test_graph
# or:
python3 Libs/test_graph.py
```

Run the example demonstration:
```bash
python3 Libs/example.py
```
