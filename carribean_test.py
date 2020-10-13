"""
"""

import unittest
import os.path

import numpy as np

import carribean
from carribean.points_grid import PointsGridGraph, four_points_connectivity, eight_points_connectivity
from carribean.carribean import get_best_island


class PointsGridGraphTest(unittest.TestCase):
    """
    Simple test case for the PointsGridGraph and get_best_island
    """

    def setUp(self):
        self.input = np.genfromtxt(os.path.join(os.path.dirname(carribean.__file__), 'test_data.csv'), delimiter=',')
        self.input = self.input.T


    def test_four_points_connectivity(self):
        """
        tests that four point connectivity provides the right number of components
        """

        connectivity = four_points_connectivity
        graph = PointsGridGraph(input_map=self.input, connectivity_strategy=connectivity)
        components = graph.get_connected_components(min_component_size=1)
        self.assertEqual(7, len(components.island.unique()))
    
    def test_eight_points_connectivity(self):
        """
        tests that eight oint connectivity provides the right number of components
        """

        connectivity = eight_points_connectivity
        graph = PointsGridGraph(input_map=self.input, connectivity_strategy=connectivity)
        components = graph.get_connected_components(min_component_size=1)
        self.assertEqual(5, len(components.island.unique()))
    
    def test_min_filter(self):
        """
        tests the minimal island size filter 
        """

        connectivity = four_points_connectivity
        graph = PointsGridGraph(input_map=self.input, connectivity_strategy=connectivity)
        components = graph.get_connected_components(min_component_size=9)
        self.assertEqual(1, len(components.island.unique()))
    
    def test_get_best_island(self):
        """
        tests that get_best_island returns the right score
        """

        connectivity = four_points_connectivity
        graph = PointsGridGraph(input_map=self.input, connectivity_strategy=connectivity)
        components = graph.get_connected_components(min_component_size=2)
        score, _ = get_best_island(components)

        self.assertEqual(5.75, score)


if __name__ == '__main__':
    unittest.main()
