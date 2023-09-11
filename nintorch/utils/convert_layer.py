from typing import Callable

import torch.nn as nn

__all__ = ["convert_layer", "convert_layer2"]


# https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
def convert_layer(
    module: nn.Module,
    target_module: Callable,
    replace_module: Callable,
    replace_params: bool = True,
    *args,
    **kwargs
) -> nn.Module:
    """Convert all `target_module` layer in `module` to `replace_module`.

    Example:
    >>> model = resnet18()
    >>> model = convert_layer(model, nn.BatchNorm2d, nn.SyncBatchNorm)
    >>> model = convert_layer(model, nn.Linear, AnotherLinear)
    >>> model = convert_layer(model, nn.Conv2d, nn.ConvTranspose2d)
    """
    assert isinstance(replace_params, bool)
    module_output = module

    if isinstance(module, target_module):
        if isinstance(module, nn.modules.conv._ConvNd):
            try:
                module_output = replace_module(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    *args,
                    **kwargs,
                )
            except TypeError:
                module_output = replace_module(*args, **kwargs)

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias

        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            try:
                module_output = replace_module(
                    module.num_features,
                    module.eps,
                    module.momentum,
                    module.affine,
                    module.track_running_stats,
                    *args,
                    **kwargs,
                )
            except TypeError:
                module_output = replace_module(*args, **kwargs)

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var

        elif isinstance(module, nn.Linear):
            module_output = replace_module(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias

        elif isinstance(module, nn.Module):
            try:
                module_output = replace_module(inplace=module.inplace, *args, **kwargs)
            except (TypeError, AttributeError):
                # A problem with third party built-in activation.
                # Some activation may not contains the inplace attribute.
                # If exception raised, recreates module without an inplace argument.
                module_output = replace_module(*args, **kwargs)

        else:
            raise NotImplementedError()

    if hasattr(module, "qconfig"):
        module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(
            name,
            convert_layer(
                child, target_module, replace_module, replace_params, *args, **kwargs
            ),
        )
    del module
    return module_output


def convert_layer2(
    module: nn.Module,
    target_module: Callable,
    replace_module: Callable,
    replace_params: bool = True,
    copy: bool = False,
    *args,
    **kwargs
) -> nn.Module:
    """Convert all `target_module` layer in `module` to `replace_module`.

    Args and kwargs will broadcast to all modules.

    Arguments:
        copy: copy all `args` and `kwargs` to break shallow copies.

    Example:
    >>> model = resnet18()
    >>> model = convert_layer(model, nn.BatchNorm2d, nn.SyncBatchNorm)
    >>> model = convert_layer(model, nn.Linear, AnotherLinear)
    >>> model = convert_layer(model, nn.Conv2d, nn.ConvTranspose2d)
    """
    assert isinstance(replace_params, bool)
    module_output = module

    if isinstance(module, target_module):
        if copy:
            copied_args = []
            for arg in args:
                arg = deepcopy(arg)
                copied_args.append(arg)

            copied_kwargs = {}
            for k, v in kwargs.items():
                copied_kwargs[k] = deepcopy(v)

            args = copied_args
            kwargs = copied_kwargs

        if isinstance(module, nn.modules.conv._ConvNd):
            try:
                module_output = replace_module(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    *args,
                    **kwargs,
                )
            except TypeError:
                module_output = replace_module(*args, **kwargs)

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias

        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            try:
                module_output = replace_module(
                    module.num_features,
                    module.eps,
                    module.momentum,
                    module.affine,
                    module.track_running_stats,
                    *args,
                    **kwargs,
                )
            except TypeError:
                module_output = replace_module(*args, **kwargs)

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var

        elif isinstance(module, nn.Linear):
            module_output = replace_module(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                *args,
                **kwargs,
            )

            if replace_params:
                module_output.weight = module.weight
                module_output.bias = module.bias

        elif isinstance(module, nn.Module):
            try:
                module_output = replace_module(inplace=module.inplace, *args, **kwargs)
            except (TypeError, AttributeError):
                # A problem with third party built-in activation.
                # Some activation may not contains the inplace attribute.
                # If exception raised, recreates module without an inplace argument.
                module_output = replace_module(*args, **kwargs)

        else:
            raise NotImplementedError()

    if hasattr(module, "qconfig"):
        module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(
            name,
            convert_layer2(
                child,
                target_module,
                replace_module,
                replace_params,
                copy,
                *args,
                **kwargs,
            ),
        )
    del module
    return module_output


if __name__ == "__main__":
    from torchvision.models import resnet18

    class AnotherLinear(nn.Linear):
        """Just for replace Linear to AnotherLinear for debug."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    model = resnet18()
    model = convert_layer(model, nn.BatchNorm2d, nn.SyncBatchNorm, True)
    model = convert_layer(model, nn.Linear, AnotherLinear, True)
    model = convert_layer(model, nn.Conv2d, nn.ConvTranspose2d, True)

    # print(model)

    print(model.fc.weight)
    print("Convert `BatchNorm2d -> SyncBatchNorm`")
    print("Convert `Linear -> AnotherLinear`")
    print("Convert `Conv2d -> Conv2dTransposed2d`")
    print(model.fc.weight)
