# https://github.com/kuangliu/pytorch-cifar
from typing import Any, Optional

from .densenet import *
from .dla import *
from .dla_simple import *
from .dpn import *
from .efficientnet import *
from .googlenet import *
from .lenet import *
from .mobilenet import *
from .mobilenetv2 import *
from .pnasnet import *
from .preact_resnet import *
from .regnet import *
from .resnet import *

# https://github.com/akamaster/pytorch_resnet_cifar10
from .resnet_cifar import *
from .resnext import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .vgg import *

__all__ = [
    'VGG',
    'DenseNet121',
    'DenseNet161',
    'DenseNet169',
    'DenseNet201',
    'EfficientNetB0',
    'GoogLeNet',
    'MobileNet',
    'MobileNetV2',
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'ResNeXt29_2x64d',
    'ResNeXt29_4x64d',
    'ResNeXt29_8x64d',
    'ResNeXt29_32x4d',
    'SENet18',
    'ShuffleNetG2',
    'ShuffleNetG3',
    'ShuffleNetV2',
    'resnet20',
    'resnet32',
    'resnet44',
    'resnet56',
    'resnet110',
    'resnet1202',
    'construct_model_cifar',
]


def construct_model_cifar(model_name: str, num_classes: Optional[int] = None, *args: Any, **kwargs: Any) -> nn.Module:
    """Construct model given model name.

    Arguments:
        model_name: model name to construct.
        num_classes: number of classes to replace last linear layers with.
    """
    model_name = model_name.lower()
    if model_name == 'lenet5':
        model = LeNet5(*args, **kwargs)

    elif model_name == 'vgg11':
        model = VGG('VGG11', *args, **kwargs)
    elif model_name == 'vgg13':
        model = VGG('VGG13', *args, **kwargs)
    elif model_name == 'vgg16':
        model = VGG('VGG16', *args, **kwargs)
    elif model_name == 'vgg19':
        model = VGG('VGG19', *args, **kwargs)

    elif model_name == 'resnet18':
        model = ResNet18(*args, **kwargs)
    elif model_name == 'resnet34':
        model = ResNet34(*args, **kwargs)
    elif model_name == 'resnet50':
        model = ResNet50(*args, **kwargs)
    elif model_name == 'resnet101':
        model = ResNet101(*args, **kwargs)
    elif model_name == 'resnet152':
        model = ResNet152(*args, **kwargs)

    elif model_name == 'resnet20':
        model = resnet20(*args, **kwargs)
    elif model_name == 'resnet32':
        model = resnet32(*args, **kwargs)
    elif model_name == 'resnet44':
        model = resnet44(*args, **kwargs)
    elif model_name == 'resnet56':
        model = resnet56(*args, **kwargs)
    elif model_name == 'resnet110':
        model = resnet110(*args, **kwargs)
    elif model_name == 'resnet1202':
        model = resnet1202(*args, **kwargs)

    elif model_name == 'googlelenet':
        model = GoogLeNet(*args, **kwargs)
    elif model_name == 'densenet121':
        model = DenseNet121(*args, **kwargs)
    elif model_name == 'densenet161':
        model = DenseNet161(*args, **kwargs)
    elif model_name == 'densenet169':
        model = DenseNet169(*args, **kwargs)
    elif model_name == 'densenet201':
        model = DenseNet201(*args, **kwargs)

    elif model_name == 'efficientnetb0':
        model = EfficientNetB0(*args, **kwargs)

    elif model_name == 'mobilenet':
        model = MobileNet(*args, **kwargs)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(*args, **kwargs)

    elif model_name == 'resnext29_2x64d':
        model = ResNeXt29_2x64d(*args, **kwargs)
    elif model_name == 'resnext29_4x64d':
        model = ResNeXt29_4x64d(*args, **kwargs)
    elif model_name == 'resnext29_8x64d':
        model = ResNeXt29_8x64d(*args, **kwargs)
    elif model_name == 'resnext29_32x4d':
        model = ResNeXt29_32x4d(*args, **kwargs)
    elif model_name == 'shufflenetg2':
        model = ShuffleNetG2(*args, **kwargs)
    elif model_name == 'shufflenetg3':
        model = ShuffleNetG3(*args, **kwargs)
    elif model_name == 'shufflenetv2':
        model = ShuffleNetV2(*args, **kwargs)

    elif model_name == 'senet18':
        model = SENet18(*args, **kwargs)
    else:
        raise NotImplementedError(f'Your models `{model_name.lower(*args, **kwargs)}` does not supported.')

    if num_classes is not None:
        try:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        except AttributeError:
            model.linear = nn.Linear(model.linear.in_features, num_classes)
    return model
