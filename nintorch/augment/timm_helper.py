import math

import albumentations as A
import cv2
import timm
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from timm.data.constants import DEFAULT_CROP_PCT
from timm.data.transforms_factory import create_transform

STR2INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczo": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


# TODO: support is_train
def convert_timm_conf_albu(model: nn.Module, is_train: bool = False) -> A.Compose:
    """Convert `timm` model configurations to albumentations transforms."""
    if is_train:
        raise NotImplementedError('`is_train = True` is still not support yet.')

    data_conf = timm.data.resolve_model_data_config(model)
    mean, std = data_conf["mean"], data_conf["std"]

    input_size = data_conf["input_size"]
    crop_pct = data_conf["crop_pct"]
    interpolation = data_conf["interpolation"]
    interpolation = STR2INTERP[interpolation]

    crop_pct = crop_pct or DEFAULT_CROP_PCT
    scale1 = math.floor(input_size[1] * (1.0 / crop_pct))
    scale2 = math.floor(input_size[2] * (1.0 / crop_pct))
    scale_size = (scale1, scale2)

    crop_mode = data_conf["crop_mode"]
    if crop_mode == "squash":
        tfl = [
            A.Resize(scale_size[0], scale_size[0], interpolation),
            A.CenterCrop(input_size[1], input_size[2]),
        ]
    elif crop_mode == "border":
        fill = [round(255.0 * m) for m in mean]
        amax = max(scale_size)
        tfl = [
            A.LongestMaxSize(amax, interpolation=interpolation),
            A.PadIfNeeded(min_height=amax, min_width=amax, border_mode=0, value=fill),
        ]
    else:
        # crop_mode == `center`
        if scale_size[0] == scale_size[1]:
            tfl = [A.Resize(scale_size[0], scale_size[1], interpolation=interpolation)]
        else:
            amax = max(scale_size)
            tfl = [A.LongestMaxSize(amax, interpolation=interpolation)]
        tfl += [A.CenterCrop(input_size[1], input_size[2])]
    tfl += [A.Normalize(mean, std, max_pixel_value=255), ToTensorV2()]
    compose = A.Compose(tfl)
    return compose


if __name__ == "__main__":
    model = timm.create_model(
        "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384", False
    )
    # model = timm.create_model("resnet152.a1h_in1k", False)
    # model = timm.create_model("vit_base_patch16_224.augreg_in21k", False)
    # model = timm.create_model("swin_base_patch4_window7_224.ms_in22k_ft_in1k", False)
    # model = timm.create_model("beit_large_patch16_384.in22k_ft_in22k_in1k", False)
    # model = timm.create_model("levit_256.fb_dist_in1k", False)
    # model = timm.create_model('efficientnetv2_rw_t.ra2_in1k', False)

    data_conf = timm.data.resolve_model_data_config(model)
    transform = create_transform(**data_conf)
    print(transform)

    albu_transform = convert_timm_conf_albu(model)
    print(albu_transform)
