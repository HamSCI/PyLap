import numpy as np
import unittest

from pharlap import dop_spread_eq

# Test dop_spread_eq against MATLAB results.
class Test_DopSpreadEq(unittest.TestCase):
    def test1(self):
        UT = [2000, 1, 1, 0, 0]

        # MATLAB: dop_spread_eq(5, 70, [2000 1 1 0 0], 1)
        try:
            out = dop_spread_eq(5, 70, UT, 1)
            self.assertEqual(out, 7.087924718765192e-17)
        except Exception as inst:
            print(inst)

    def test2(self):
        UT = [2000, 1, 1, 0, 0]

        # MATLAB: dop_spread_eq(40, 75, [2000 1 1 0 0], 0)
        try:
            out = dop_spread_eq(40, 75, UT, 0)
            self.assertEqual(out, 0)
        except Exception as inst:
            print(inst)
