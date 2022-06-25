import numpy as np
import unittest

from pharlap import iri2016

class Test_Iri2016(unittest.TestCase):
    def setUp(self):
        self.expected_iono_001 = np.loadtxt('./data/iri2016_iono_001.txt')
        self.expected_extra_001 = np.loadtxt('./data/iri2016_extra_001.txt')

        self.expected_iono_002 = np.loadtxt('./data/iri2016_iono_002.txt')
        self.expected_extra_002 = np.loadtxt('./data/iri2016_extra_002.txt')

    def test_base(self):
        # MATLAB: iri2016(45, 75, 1, [2000 1 1 0 0])
        #try:
        [iono, extra] = iri2016(45, 75, 1, [2000, 1, 1, 0, 0])
        print('IRI EXTRA PRINT')
        print(extra)
        print('IRI SELF FILE PRINT')
        print(self.expected_extra_001)
        np.testing.assert_array_almost_equal(extra, self.expected_extra_001)
        #except Exception as inst:
        #    print(inst)

    def test_heights(self):
        # MATLAB: iri2016(45, 75, 1, [2000 1 1 0 0], 100, 50, 5)
        # try:
        [iono, extra] = iri2016(45, 75, 1, [2000, 1, 1, 0, 0], 100, 50, 5)

        np.testing.assert_array_almost_equal(iono, self.expected_iono_001)
        np.testing.assert_array_almost_equal(extra, self.expected_extra_001)
        # except Exception as inst:
        #     print(inst)

    def test_no_r12(self):
        # MATLAB: iri2016(45, 75, -1, [2000 1 1 0 0], 100, 50, 5)
        #try:
        [iono, extra] = iri2016(45, 75, -1, [2000, 1, 1, 0, 0], 100, 50, 5)

        np.testing.assert_array_almost_equal(iono, self.expected_iono_002)
        np.testing.assert_array_almost_equal(extra, self.expected_extra_002)
        #except Exception as inst:
         #   print(inst)
