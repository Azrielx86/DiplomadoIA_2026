"""
Comprehensive unit tests for the generic graph library.
"""

import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

# Ensure Libs can be imported regardless of execution directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Libs import AStarResult, Edge, Graph, Node, a_star


@dataclass
class Point2D:
    x: float
    y: float


class TestNode(unittest.TestCase):
    def test_node_creation_and_attributes(self):
        node = Node[int]("A", 42)
        self.assertEqual(node.id, "A")
        self.assertEqual(node.data, 42)
        self.assertEqual(node.value, 42)

        # Test value setter
        node.value = 100
        self.assertEqual(node.data, 100)

    def test_auto_id_generation(self):
        node1 = Node[str]()
        node2 = Node[str]()
        self.assertTrue(node1.id.startswith("node_"))
        self.assertTrue(node2.id.startswith("node_"))
        self.assertNotEqual(node1.id, node2.id)

    def test_node_neighbors(self):
        node = Node[str]("N1", "start")
        node.add_neighbor("N2", 3.5)
        node.add_neighbor("N3", 7.0)

        self.assertTrue(node.has_neighbor("N2"))
        self.assertTrue(node.has_neighbor("N3"))
        self.assertFalse(node.has_neighbor("N4"))
        self.assertEqual(node.get_distance_to("N2"), 3.5)
        self.assertEqual(node.get_distance_to("N3"), 7.0)
        self.assertEqual(node.degree, 2)

        removed = node.remove_neighbor("N2")
        self.assertTrue(removed)
        self.assertFalse(node.has_neighbor("N2"))
        self.assertEqual(node.degree, 1)

    def test_node_equality_and_hash(self):
        n1 = Node[int]("A", 10)
        n2 = Node[int]("A", 20)  # Same ID
        n3 = Node[int]("B", 10)

        self.assertEqual(n1, n2)
        self.assertNotEqual(n1, n3)
        self.assertEqual(n1, "A")

        s = {n1, n2, n3}
        self.assertEqual(len(s), 2)


class TestEdge(unittest.TestCase):
    def test_edge_undirected(self):
        n1 = Node[str]("A")
        n2 = Node[str]("B")
        e1 = Edge(n1, n2, distance=5.0, directed=False)
        e2 = Edge(n2, n1, distance=5.0, directed=False)

        self.assertEqual(e1, e2)
        self.assertEqual(e1.distance, 5.0)
        self.assertEqual(e1.weight, 5.0)

    def test_edge_directed(self):
        n1 = Node[str]("A")
        n2 = Node[str]("B")
        e1 = Edge(n1, n2, distance=5.0, directed=True)
        e2 = Edge(n2, n1, distance=5.0, directed=True)

        self.assertNotEqual(e1, e2)


class TestGraph(unittest.TestCase):
    def test_add_and_get_nodes(self):
        g = Graph[str]()
        node_a = g.add_node("A", "Data A")
        node_b = g.add_node("B", "Data B")

        self.assertEqual(g.get_node("A"), node_a)
        self.assertEqual(g.find_node("B"), node_b)
        self.assertIsNone(g.get_node("C"))

        self.assertTrue("A" in g)
        self.assertTrue(node_a in g)
        self.assertFalse("C" in g)
        self.assertEqual(len(g), 2)

    def test_add_node_updating_data(self):
        g = Graph[int]()
        g.add_node("A", 10)
        self.assertEqual(g["A"].data, 10)

        # Update data with same ID
        g.add_node("A", 25)
        self.assertEqual(g["A"].data, 25)

    def test_remove_node(self):
        g = Graph[str]()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", distance=4.0)

        self.assertTrue(g.has_edge("A", "B"))
        removed = g.remove_node("B")
        self.assertTrue(removed)
        self.assertFalse("B" in g)
        self.assertFalse(g.has_edge("A", "B"))
        self.assertEqual(len(g), 1)

    def test_find_nodes(self):
        g = Graph[int]()
        g.add_node("n1", 10)
        g.add_node("n2", 20)
        g.add_node("n3", 30)
        g.add_node("n4", 20)

        by_val = g.find_nodes_by_value(20)
        self.assertEqual(len(by_val), 2)
        self.assertEqual({n.id for n in by_val}, {"n2", "n4"})

        greater_than_15 = g.find_nodes_by_data(lambda d: d is not None and d > 15)
        self.assertEqual(len(greater_than_15), 3)

        matching_id = g.find_nodes(lambda n: n.id == "n1")
        self.assertEqual(len(matching_id), 1)
        self.assertEqual(matching_id[0].id, "n1")

    def test_edges_and_distances(self):
        g = Graph[None]()
        # Auto-creation of nodes when adding edge
        edge = g.add_edge("X", "Y", distance=12.5)

        self.assertEqual(len(g), 2)
        self.assertTrue(g.has_edge("X", "Y"))
        self.assertTrue(g.has_edge("Y", "X"))  # Undirected
        self.assertEqual(g.get_edge_distance("X", "Y"), 12.5)
        self.assertEqual(g.get_edge_distance("Y", "X"), 12.5)

        # Update distance
        g.set_edge_distance("X", "Y", 18.0)
        self.assertEqual(g.get_edge_distance("X", "Y"), 18.0)
        self.assertEqual(g.get_edge_distance("Y", "X"), 18.0)

        # Remove edge
        self.assertTrue(g.remove_edge("X", "Y"))
        self.assertFalse(g.has_edge("X", "Y"))
        self.assertFalse(g.has_edge("Y", "X"))

    def test_directed_graph(self):
        dg = Graph[str](directed=True)
        dg.add_edge("1", "2", distance=3.0)

        self.assertTrue(dg.has_edge("1", "2"))
        self.assertFalse(dg.has_edge("2", "1"))
        self.assertEqual(dg.get_edge_distance("1", "2"), 3.0)
        self.assertIsNone(dg.get_edge_distance("2", "1"))

    def test_get_neighbors(self):
        g = Graph[str]()
        g.add_edge("A", "B", distance=2.0)
        g.add_edge("A", "C", distance=4.5)

        neighbors = g.get_neighbors("A")
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[g.get_node("B")], 2.0)
        self.assertEqual(neighbors[g.get_node("C")], 4.5)

        neighbor_nodes = g.get_neighbor_nodes("A")
        self.assertEqual({n.id for n in neighbor_nodes}, {"B", "C"})


class TestAStar(unittest.TestCase):
    def test_a_star_simple(self):
        g = Graph[str]()
        g.add_edge("A", "B", distance=1.0)
        g.add_edge("B", "C", distance=2.0)
        g.add_edge("A", "C", distance=5.0)

        result = g.a_star("A", "C")
        self.assertIsNotNone(result)
        self.assertEqual(result.path_ids, ["A", "B", "C"])
        self.assertEqual(result.total_distance, 3.0)

        # Test tuple unpacking
        path, distance = result
        self.assertEqual([n.id for n in path], ["A", "B", "C"])
        self.assertEqual(distance, 3.0)

    def test_a_star_with_custom_heuristic(self):
        # 2D Grid with Euclidean coordinates
        # A(0,0) -> B(1,2) -> C(4,6) (Distance: sqrt(5) + 5 = 2.236 + 5 = 7.236)
        # A(0,0) -> D(3,0) -> C(4,6) (Distance: 3 + sqrt(1 + 36) = 3 + 6.082 = 9.082)
        g = Graph[Point2D]()
        g.add_node("A", Point2D(0.0, 0.0))
        g.add_node("B", Point2D(1.0, 2.0))
        g.add_node("C", Point2D(4.0, 6.0))
        g.add_node("D", Point2D(3.0, 0.0))

        def euclidean_dist(p1: Point2D, p2: Point2D) -> float:
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        g.add_edge("A", "B", distance=euclidean_dist(g["A"].data, g["B"].data))
        g.add_edge("B", "C", distance=euclidean_dist(g["B"].data, g["C"].data))
        g.add_edge("A", "D", distance=euclidean_dist(g["A"].data, g["D"].data))
        g.add_edge("D", "C", distance=euclidean_dist(g["D"].data, g["C"].data))

        # Heuristic estimating straight line distance to goal
        def heuristic(curr: Node[Point2D], goal: Node[Point2D]) -> float:
            assert curr.data is not None and goal.data is not None
            return euclidean_dist(curr.data, goal.data)

        result = g.a_star("A", "C", heuristic=heuristic)
        self.assertIsNotNone(result)
        self.assertEqual(result.path_ids, ["A", "B", "C"])
        expected_distance = euclidean_dist(Point2D(0, 0), Point2D(1, 2)) + euclidean_dist(
            Point2D(1, 2), Point2D(4, 6)
        )
        self.assertAlmostEqual(result.total_distance, expected_distance, places=4)

    def test_a_star_single_arg_heuristic(self):
        g = Graph[int]()
        g.add_edge("A", "B", distance=1.0)
        g.add_edge("B", "C", distance=1.0)

        # Heuristic accepting only current node
        result = g.a_star("A", "C", heuristic=lambda n: 0.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.path_ids, ["A", "B", "C"])

    def test_a_star_same_start_and_goal(self):
        g = Graph[str]()
        g.add_node("A", "StartGoal")
        result = g.a_star("A", "A")

        self.assertIsNotNone(result)
        self.assertEqual(result.path_ids, ["A"])
        self.assertEqual(result.total_distance, 0.0)

    def test_a_star_unreachable(self):
        g = Graph[str]()
        g.add_node("A")
        g.add_node("B")  # Disconnected

        result = g.a_star("A", "B")
        self.assertIsNone(result)

    def test_a_star_missing_node_raises(self):
        g = Graph[str]()
        g.add_node("A")
        with self.assertRaises(KeyError):
            g.a_star("A", "NonExistent")

    def test_a_star_directed_respects_orientation(self):
        dg = Graph[str](directed=True)
        dg.add_edge("A", "B", distance=1.0)

        result_forward = dg.a_star("A", "B")
        self.assertIsNotNone(result_forward)
        self.assertEqual(result_forward.path_ids, ["A", "B"])

        result_backward = dg.a_star("B", "A")
        self.assertIsNone(result_backward)


class TestPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_circular_layout(self):
        g = Graph[str]()
        g.add_edge("A", "B", distance=2.0)
        g.add_edge("B", "C", distance=3.5)

        ax = g.plot(show=False)
        self.assertIsNotNone(ax)

    def test_plot_with_coordinates(self):
        g = Graph[Point2D]()
        g.add_node("A", Point2D(0.0, 0.0))
        g.add_node("B", Point2D(3.0, 4.0))
        g.add_edge("A", "B", distance=5.0)

        ax = g.plot(show=False)
        self.assertIsNotNone(ax)

    def test_plot_with_highlight_path(self):
        g = Graph[int]()
        g.add_edge("1", "2", distance=1.0)
        g.add_edge("2", "3", distance=2.0)
        result = g.a_star("1", "3")

        ax = g.plot(highlight_path=result, show=False)
        self.assertIsNotNone(ax)

    def test_plot_save_to_file(self):
        import os
        import tempfile

        g = Graph[str]()
        g.add_edge("X", "Y", distance=1.5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            ax = g.plot(show=False, save_path=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_directed(self):
        dg = Graph[str](directed=True)
        dg.add_edge("A", "B", distance=4.0)
        ax = dg.plot(show=False)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
