import torch
from torch import nn
from torch.nn.utils import prune

from nintorch import count_sparse_module


def test_count_sparse():
    with torch.inference_mode():
        model = nn.Linear(1, 10, bias=False)
        model.weight = nn.Parameter(torch.zeros(1, 10))
        sparse = count_sparse_module(model)
        assert sparse == 1.0

        model = nn.Linear(1, 10, bias=False)
        weight = torch.zeros(1, 10)
        weight[0, 0] = 1.0
        model.weight = nn.Parameter(weight)
        sparse = count_sparse_module(model)
        torch.testing.assert_close(sparse, 0.9)

        model = nn.Linear(1, 10, bias=True)
        model.weight = nn.Parameter(torch.zeros(1, 10))
        sparse_dict = count_sparse_module(model, bias=True, return_layers=True)
        torch.testing.assert_close(sparse_dict['weight'], 1.0)
        torch.testing.assert_close(sparse_dict['bias'], 0.0)

        model = nn.Linear(1, 10, bias=True)
        prune.l1_unstructured(model, 'weight', 0.5)
        sparse = count_sparse_module(model, bias=True, return_layers=False)
        torch.testing.assert_close(sparse, 0.5)
