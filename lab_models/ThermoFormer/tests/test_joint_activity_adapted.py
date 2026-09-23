import unittest

import torch

from src.thermoformer.baselines.machine_learning.joint_activity_adapted import (
    JointActivityResidualHead,
)
from src.thermoformer.baselines.machine_learning.protocols import benchmark_for_baseline
from src.thermoformer.baselines.machine_learning.schema import BASELINE_CAPABILITIES


class JointActivityAdaptedTests(unittest.TestCase):
    def test_zero_initialized_head_preserves_official_prediction(self) -> None:
        model = JointActivityResidualHead(hidden_dimension=8, dropout=0.0).eval()
        base = torch.tensor([[0.1, -0.2], [0.4, 0.3]])
        x = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        temperature = torch.tensor([[300.0], [350.0]])
        torch.testing.assert_close(model(base, temperature, x), base)

    def test_shared_head_is_permutation_equivariant(self) -> None:
        torch.manual_seed(7)
        model = JointActivityResidualHead(hidden_dimension=8, dropout=0.0).eval()
        torch.nn.init.normal_(model.network[-1].weight)
        base = torch.tensor([[0.1, -0.2, 0.4]])
        x = torch.tensor([[0.2, 0.3, 0.5]])
        temperature = torch.tensor([[330.0]])
        permutation = torch.tensor([2, 0, 1])
        expected = model(base, temperature, x)[:, permutation]
        actual = model(base[:, permutation], temperature, x[:, permutation])
        torch.testing.assert_close(actual, expected)



if __name__ == "__main__":
    unittest.main()
