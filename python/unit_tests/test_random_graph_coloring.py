import unittest
import os
import sys
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.proximal_contact.blocking as BLOCKING
import rainbow.util.test_tools as TEST


class TestBlocking(unittest.TestCase):

    def setUp(self):
        self.K = 5
        self.G = nx.Graph()
        self.G.add_nodes_from(np.arange(self.K))

    def test_random_graph_coloring(self):
        # Create a graph: 0-1-2-3-4, here the number is the node id.
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        color_groups = BLOCKING.random_graph_coloring(self.G)

        # Check if the color groups are valid
        # The color groups should be: {1,2,4}, {1, 3}
        expect_color_groups = {0: [0, 2, 4], 1: [1, 3]}
        for k, v in color_groups.items():
            self.assertIn(k, expect_color_groups)
            TEST.is_array_equal(v, expect_color_groups[k])


if __name__ == "__main__":
    unittest.main()