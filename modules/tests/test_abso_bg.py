from unittest.signals import installHandler
import numpy as np
import unittest
import sys
import pharlap.abso_bg as abso_bg

# Test abso_bg against MATLAB results.
class Test_AbsoBg(unittest.TestCase):
    def setUp(self):
        self.result_001 = np.loadtxt('./data/abso_bg_001.txt')
        self.result_002 = np.loadtxt('./data/abso_bg_002.txt')

    def test_x_mode(self):
        lats = np.array([40., 45.])
        lons = np.array([75., 80.])
        elevs = np.array([45., 50.])
        freqs = np.array([20., 25.])
        UT = [2020, 1, 1, 12, 0]

        # MATLAB: abso_bg([40 45], [75 80], [45 50], [20 25], [2020 1 1 12 0], 1, 0)
        try:
            out = abso_bg(lats, lons, elevs, freqs, UT, 1, 0)
            np.testing.assert_array_almost_equal(out, self.result_001)
        except Exception as inst:
            print(inst)

    def test_o_mode(self):
        lats = np.array([40., 45.])
        lons = np.array([75., 80.])
        elevs = np.array([45., 50.])
        freqs = np.array([20., 25.])
        UT = [2020, 1, 1, 12, 0]

        # MATLAB: abso_bg([40 45], [75 80], [45 50], [20 25], [2020 1 1 12 0], 1, 1)
        try:
            out = abso_bg(lats, lons, elevs, freqs, UT, 1, 1)
            np.testing.assert_array_almost_equal(out, self.result_002)
        except Exception as inst:
            print(inst)
