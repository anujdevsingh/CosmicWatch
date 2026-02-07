import math
import unittest

import numpy as np

from conjunction.pc import pc_circle_grid, pc_circle_monte_carlo


class TestPc(unittest.TestCase):
    def test_isotropic_zero_mean_closed_form_match(self):
        sigma = 0.25
        cov = [[sigma * sigma, 0.0], [0.0, sigma * sigma]]
        r = 0.5

        expected = 1.0 - math.exp(-(r * r) / (2.0 * sigma * sigma))

        grid = pc_circle_grid([0.0, 0.0], cov, r, n_theta=720, n_r=600).pc
        mc = pc_circle_monte_carlo([0.0, 0.0], cov, r, n=200_000, seed=1).pc

        self.assertAlmostEqual(grid, expected, delta=5e-3)
        self.assertAlmostEqual(mc, expected, delta=8e-3)

    def test_shifted_mean_reduces_probability(self):
        cov = [[0.1, 0.0], [0.0, 0.1]]
        r = 0.25
        pc0 = pc_circle_grid([0.0, 0.0], cov, r, n_theta=360, n_r=400).pc
        pc1 = pc_circle_grid([0.5, 0.0], cov, r, n_theta=360, n_r=400).pc
        self.assertGreater(pc0, pc1)

    def test_cov_must_be_pd(self):
        cov = [[1.0, 2.0], [2.0, 1.0]]
        with self.assertRaises(Exception):
            pc_circle_grid([0.0, 0.0], cov, 1.0)


if __name__ == "__main__":
    unittest.main()
