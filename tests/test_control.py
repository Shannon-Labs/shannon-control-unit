#!/usr/bin/env python3
import math
import unittest

from shannon_control.control import update_lambda


class TestSCUController(unittest.TestCase):
    def setUp(self):
        self.kw = dict(Kp=0.8, Ki=0.15, deadband=0.0, lmin=1e-4, lmax=2.0)

    def test_negative_feedback(self):
        # Start with nominal values
        lmbda, I, S_hat = 1.0, 0.0, 0.0
        # S too high -> lambda should increase
        l1, I1, _ = update_lambda(lmbda, 0.02, 0.01, I, **self.kw)
        self.assertGreater(l1, lmbda)
        # S too low -> lambda should decrease
        l2, I2, _ = update_lambda(lmbda, 0.005, 0.01, I, **self.kw)
        self.assertLess(l2, lmbda)

    def test_bounds_and_integral(self):
        lmbda, I, _ = update_lambda(2.0, 0.02, 0.01, 0.2, **self.kw)
        self.assertLessEqual(lmbda, 2.0 + 1e-9)
        self.assertLessEqual(I, 0.2 + 1e-6)


if __name__ == "__main__":
    unittest.main()

