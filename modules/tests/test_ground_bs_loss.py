import numpy as np
import unittest

from pharlap import ground_bs_loss

# Tests ground_bs_loss against MATLAB results.
class Test_GroundBsLoss(unittest.TestCase):
    def test_one_input(self):
        lat = np.array([40.])
        lon = np.array([75.])

        try:
            out = ground_bs_loss(lat, lon)
            self.assertEqual(out, 29.030899869919438)
        except Exception as inst:
            print(inst)

    def test_multiple_input(self):
        lats = np.array([45., 75.])
        lons = np.array([45., 80.])

        # MATLAB: ground_bs_loss([45 75], [45 80])
        try:
            out = ground_bs_loss(lats, lons)

            self.assertEqual(out[0], 29.030899869919438)
            self.assertEqual(out[1], 26.020599913279625)
        except Exception as inst:
            print(inst)
