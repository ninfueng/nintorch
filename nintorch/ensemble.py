from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from nintorch import infer_mode

__all__ = ['VotingEnsemble']


class VotingEnsemble(nn.Module):
    """Ensemble by averaging or voting output logits."""

    def __init__(self, models: Sequence[nn.Module], mode: str = 'soft') -> None:
        super().__init__()
        assert mode in (
            'soft',
            'hard',
        ), f'`mode` should be in `(soft, hard)`. Your: {mode}.'
        self.models = nn.ModuleList(models)
        self.mode = mode

    @infer_mode()
    def forward(self, input: Tensor) -> Tensor:
        if self.mode == 'soft':
            output = self._forward_avg(input)
        elif self.mode == 'hard':
            output = self._forward_vote(input)
        else:
            raise ValueError(
                f'Support `mode` only `soft` or `hard`, Your: {self.mode}.'
            )
        return output

    @infer_mode()
    def _forward_avg(self, input: Tensor) -> Tensor:
        preds = [model(input) for model in self.models]
        preds = torch.stack(preds, dim=-1)
        return preds.mean(dim=-1)

    @infer_mode()
    def _forward_vote(self, input: Tensor) -> Tensor:
        votes = [model(input).argmax(dim=-1) for model in self.models]
        votes = torch.stack(votes, dim=-1)
        votes, _ = torch.mode(votes, dim=-1)
        return votes


if __name__ == '__main__':
    ens = VotingEnsemble([nn.Linear(10, 1), nn.Linear(10, 1)])
    input = torch.randn(2, 10)
    output = ens(input)
    print(output.shape)
