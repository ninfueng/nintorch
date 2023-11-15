from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['VotingEnsemble']


class VotingEnsemble(nn.Module):
    """Ensemble by averaging all output logits."""

    def __init__(self, models: Sequence[nn.Module], mode: str = 'soft') -> None:
        super().__init__()
        assert mode in [
            'soft',
            'hard',
        ], f'`mode` should be in `(soft, hard)`. Your: {mode}.'
        self.models = nn.ModuleList(models)
        self.mode = mode

    @torch.inference_mode()
    def forward(self, input: Tensor) -> Tensor:
        if self.mode == 'soft':
            output = self._forward_avg(input)
        elif self.mode == 'hard':
            output = self._forward_vote(input)
        else:
            raise ValueError(f'Support `mode` only `soft` or `hard`, Your: {self.mode}.')
        return output

    @torch.inference_mode()
    def _forward_avg(self, input: Tensor) -> Tensor:
        mean = 0.0
        for model in self.models:
            pred = model(input)
            mean += pred
        mean /= len(self.models)
        return mean

    @torch.inference_mode()
    def _forward_vote(self, input: Tensor) -> Tensor:
        votes = []
        for model in self.models:
            pred = model(input)
            vote = pred.argmax(dim=-1)
            votes.append(vote)
        votes = torch.stack(votes, dim=-1)
        votes, _ = torch.mode(votes, dim=-1)
        return votes
