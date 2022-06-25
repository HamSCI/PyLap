import numpy as np
import unittest

from pharlap import ground_fs_loss

# Test ground_fs_loss against MATLAB results.
class Test_GroundFsLoss(unittest.TestCase):
    def setUp(self):
        self.expected_001 = np.loadtxt('./data/ground_fs_loss_001.txt')

    def test_one_input(self):
        lat = np.array([40.])
        lon = np.array([75.])
        elev = np.array([45.])
        freq = np.array([25.])

        # MATLAB: ground_fs_loss([40], [75], [45], [25])
        try:
            out = ground_fs_loss(lat, lon, elev, freq)

            self.assertEqual(out, 4.591796221054666)
        except Exception as inst:
            print(inst)

    def test_multiple_input(self):
        lats = np.array([40., 60.])
        lons = np.array([75., 80.])
        elevs = np.array([45., 65.])
        freqs = np.array([20., 25.])

        # MATLAB: ground_fs_loss([40 60], [75 80], [45 65], [20 25])
        try:
            out = ground_fs_loss(lats, lons, elevs, freqs)

            np.testing.assert_array_almost_equal(out, self.expected_001)
        except Exception as inst:
            print(inst)
