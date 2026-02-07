import unittest

import numpy as np

from conjunction.cdm import parse_cdm_event


class TestCdm(unittest.TestCase):
    def test_parse_basic_fields(self):
        text = """
        OBJECT1_X = 1
        OBJECT1_Y = 2
        OBJECT1_Z = 3
        OBJECT1_X_DOT = 0.1
        OBJECT1_Y_DOT = 0.2
        OBJECT1_Z_DOT = 0.3
        OBJECT2_X = 4
        OBJECT2_Y = 5
        OBJECT2_Z = 6
        OBJECT2_X_DOT = 0.4
        OBJECT2_Y_DOT = 0.5
        OBJECT2_Z_DOT = 0.6
        OBJECT1_CRR = 1e-6
        OBJECT1_CTT = 1e-6
        OBJECT1_CNN = 1e-6
        OBJECT2_CRR = 2e-6
        OBJECT2_CTT = 2e-6
        OBJECT2_CNN = 2e-6
        HARD_BODY_RADIUS_KM = 0.01
        """
        ev = parse_cdm_event(text)
        self.assertIsNotNone(ev.primary_state_eci)
        self.assertIsNotNone(ev.secondary_state_eci)
        self.assertIsNotNone(ev.primary_pos_cov_rtn)
        self.assertIsNotNone(ev.secondary_pos_cov_rtn)
        self.assertAlmostEqual(ev.hard_body_radius_km, 0.01)
        self.assertTrue(np.allclose(ev.primary_pos_cov_rtn, ev.primary_pos_cov_rtn.T))


if __name__ == "__main__":
    unittest.main()
