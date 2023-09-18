import torch
from torch import nn
from torch.nn.utils import prune

from nintorch import count_sparse


def test_count_sparse():
    with torch.no_grad():
        model = nn.Linear(1, 10, bias=False)
        model.weight = nn.Parameter(torch.zeros(1, 10))
        sparse, _ = count_sparse(model)
        assert sparse == 1.0

        model = nn.Linear(1, 10, bias=False)
        weight = torch.zeros(1, 10)
        weight[0, 0] = 1.0
        model.weight = nn.Parameter(weight)
        sparse, _ = count_sparse(model)
        assert sparse == 0.9

        model = nn.Linear(1, 10, bias=True)
        model.weight = nn.Parameter(torch.zeros(1, 10))
        _, name_sparse = count_sparse(model, skip_bias=False)
        assert name_sparse["weight"] == 1.0
        assert name_sparse["bias"] == 0.0

        model = nn.Linear(1, 10, bias=True)
        prune.l1_unstructured(model, "weight", 0.5)
        sparse, name_sparse = count_sparse(model, skip_bias=False)
        assert sparse == 0.5
