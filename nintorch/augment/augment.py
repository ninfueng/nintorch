from albumentations.core.transforms_interface import ImageOnlyTransform


class Identity(ImageOnlyTransform):
    def apply(self, img, **_):
        return img
