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

    def test_param_bpt_normalization(self):
        """Test that ParamBPT scales inversely with tokens_per_epoch."""
        from shannon_control.control import calculate_param_bpt
        import torch
        
        # Create a dummy model with one parameter
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
                
        model = DummyModel()
        
        # Calculate with N=1000
        bpt_1k = calculate_param_bpt(model, tokens_per_epoch=1000, sigma=1.0)
        
        # Calculate with N=2000
        bpt_2k = calculate_param_bpt(model, tokens_per_epoch=2000, sigma=1.0)
        
        # Should be exactly half
        self.assertAlmostEqual(bpt_2k, bpt_1k / 2.0)


if __name__ == "__main__":
    unittest.main()

