import unittest

import numpy as np

from conjunction.frames import encounter_plane, project_cov_to_encounter_plane, rtn_basis, cov_rtn_to_eci


class TestFrames(unittest.TestCase):
    def test_rtn_basis_orthonormal(self):
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 1.0])
        b = rtn_basis(r, v)
        q = b.as_matrix()
        ident = q.T @ q
        self.assertTrue(np.allclose(ident, np.eye(3), atol=1e-8))

    def test_cov_transform_symmetry(self):
        r = np.array([7000.0, 10.0, 0.0])
        v = np.array([0.0, 7.5, 0.1])
        cov_rtn = np.diag([1.0, 4.0, 9.0])
        cov_eci = cov_rtn_to_eci(cov_rtn, r, v)
        self.assertTrue(np.allclose(cov_eci, cov_eci.T, atol=1e-10))

    def test_encounter_plane_projection_shapes(self):
        r_rel = np.array([1.0, 2.0, 3.0])
        v_rel = np.array([0.1, -0.2, 0.3])
        plane = encounter_plane(r_rel, v_rel)
        cov = np.eye(3)
        cov2 = project_cov_to_encounter_plane(cov, plane)
        self.assertEqual(cov2.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
