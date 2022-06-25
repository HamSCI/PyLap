import numpy as np
import unittest

from pharlap import nrlmsise00

# Test nrlmsise00 against MATLAB results.
class Test_Nrlmsise00(unittest.TestCase):
    def setUp(self):
        self.expected_densities_001 = np.loadtxt('./data/nrlmsise00_densities_001.txt')
        self.expected_temperatures_001 = np.loadtxt('./data/nrlmsise00_temperatures_001.txt')

    def test_no_opts(self):
        lats = np.array([25., 45.])
        lons = np.array([75., 90.])
        alts = np.array([150., 250.])
        UT = [2000, 1, 1, 0, 0]

        # MATLAB: nrlmsise00([25 45], [75 90], [150 250], [2000, 1, 1, 0, 0])
        
        (out_d, out_t) = nrlmsise00(lats, lons, alts, UT)
        np.testing.assert_array_almost_equal(out_d, self.expected_densities_001)
        np.testing.assert_array_almost_equal(out_t, self.expected_temperatures_001)
