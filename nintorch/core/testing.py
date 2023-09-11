import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__file__)

__all__ = ["assert_close", "assert_model_close"]


def assert_close(a: Tensor, b: Tensor) -> None:
    """Like `torch.testing.assert_close` but with more information.

    Arguments:
        a (torch.Tensor): a tensor to find different with `b`.
        b (torch.Tensor): a tensor to find different with `a`.

    Raise:
        AssertionError: raise if some parameters in `a` != `b`.
    """
    diff = ~torch.isclose(a, b)
    diff_indices = diff.nonzero(as_tuple=False)

    if diff_indices.numel() > 0:
        diff_numel = diff.sum()
        abs_diff = (a - b).abs().amax()
        relative_diff = (a / b).abs().amax()

        error_msg = (
            f"Detect not allclose: {diff_numel} elements.\n"
            f"Max absolute diff: {abs_diff},\n"
            f"Max relative diff: {relative_diff},\n"
            f"`a` diff: {a[diff]},\n"
            f"`b` diff: {b[diff]},\n"
            f"diff indices: \n {diff_indices}"
        )
        logger.fatal(error_msg)
        assert AssertionError(error_msg)


def assert_model_close(
    a: nn.Module, b: nn.Module, check_same_name: bool = False
) -> None:
    """Assert all model parameters are same or not.

    Arguments:
        a (nn.Module): first model to check.
        b (nn.Module): second model to check with `a`.
        check_same_name (bool): whether to check the name is actually same or not.

    Raise:
        AssertionError: raise if some parameters in model do not closes.
    """
    num_param_a = len(list(a.parameters()))
    num_param_b = len(list(b.parameters()))
    assert (
        num_param_a != num_param_b
    ), f"Number of parameters is not equal: {num_param_a} != {num_param_b}."

    for idx, ((na, pa), (nb, pb)) in enumerate(
        zip(a.named_parameters(), b.named_parameters())
    ):
        if check_same_name:
            assert na == nb, f"Parameter names with {idx} are not same: {na} != {nb}."
        try:
            assert_close(pa, pb)
        except AssertionError as error_msg:
            model_error_msg = (
                f"\nDetect different `{a._get_name()}` and `{b._get_name()}`\n"
                f"At parameter name:`{na}` and `{nb}`\n"
            )
            logger.fatal(model_error_msg)
            raise AssertionError(str(error_msg) + model_error_msg)
