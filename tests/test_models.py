"""Run from the repo root:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.models.sequence import MambaBlock  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_both_models_output_one_value_per_window(self):
        x = torch.randn(5, 30, 17)
        for name in ("transformer", "mamba"):
            model = build_model(name, input_size=17, d_model=32, num_layers=2, dropout=0.1).eval()
            self.assertEqual(tuple(model(x).shape), (5,))

    def test_scan_matches_naive_recurrence(self):
        """Compare the batched scan with explicit loops over batch, channel and state."""
        torch.manual_seed(0)
        block = MambaBlock(d_model=4, d_state=3, expand=2).double()
        b, l, d, n = 2, 7, block.d_inner, 3
        u = torch.randn(b, l, d, dtype=torch.double)
        delta = F.softplus(torch.randn(b, l, d, dtype=torch.double))
        A = -torch.exp(block.A_log.double())
        B, C = torch.randn(b, l, n, dtype=torch.double), torch.randn(b, l, n, dtype=torch.double)
        D = torch.randn(d, dtype=torch.double)
        fast = block.selective_scan(u, delta, A, B, C, D).detach().numpy()

        u_, dl_, A_, B_, C_, D_ = (t.detach().numpy() for t in (u, delta, A, B, C, D))
        slow = np.zeros((b, l, d))
        for i in range(b):
            for k in range(d):
                h = np.zeros(n)
                for t in range(l):
                    h = np.exp(dl_[i, t, k] * A_[k]) * h + dl_[i, t, k] * B_[i, t] * u_[i, t, k]
                    slow[i, t, k] = (C_[i, t] * h).sum() + D_[k] * u_[i, t, k]
        np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)

    def test_block_is_causal(self):
        """Changing future inputs must not change earlier outputs."""
        torch.manual_seed(1)
        block = MambaBlock(d_model=16).eval()
        x = torch.randn(3, 20, 16)
        x2 = x.clone()
        x2[:, 12:] = torch.randn(3, 8, 16)
        with torch.no_grad():
            y, y2 = block(x), block(x2)
        torch.testing.assert_close(y[:, :12], y2[:, :12])
        self.assertFalse(torch.allclose(y[:, 12:], y2[:, 12:]))

    def test_step_size_init_range_and_gradients(self):
        torch.manual_seed(2)
        block = MambaBlock(d_model=16, dt_min=1e-3, dt_max=1e-1)
        dt = F.softplus(block.dt_proj.bias).detach()
        self.assertGreaterEqual(float(dt.min()), 1e-3 * 0.99)
        self.assertLessEqual(float(dt.max()), 1e-1 * 1.01)
        block(torch.randn(2, 10, 16)).sum().backward()
        for name, p in block.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_mamba_needs_no_optional_package(self):
        self.assertNotIn("mamba_ssm", sys.modules)
        build_model("mamba", input_size=8, d_model=16, num_layers=1)


if __name__ == "__main__":
    unittest.main()