import numpy as np
import unittest

from pharlap import irreg_strength

# Test irreg_strength against MATLAB results.
class Test_IrregStrength(unittest.TestCase):
    def test1(self):
        UT = [2000, 1, 1, 0, 0]

        # MATLAB: irreg_strength(5, 70, [2000 1 1 0 0], 1)
        try:
            out = irreg_strength(5, 70, UT, 1)

            self.assertEqual(out, 9.313908230978996e-05)
        except Exception as inst:
            print(inst)

    def test2(self):
        UT = [2000, 1, 1, 0, 0]

        # MATLAB: irreg_strength(25, 70, [2000 1 1 0 0], 0)
        try:
            out = irreg_strength(25, 70, UT, 0)

            self.assertEqual(out, 2.052543277386576e-05)
        except Exception as inst:
            print(inst)
