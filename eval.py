import argparse
import datetime
import os
import time
from pprint import pformat

import torch
import torch.nn as nn
import torchvision.models
from nincore import AttrDict
from nincore.io import load_yaml
from nincore.time import second_to_ddhhmmss
from nincore.utils import filter_warn, set_logger
from torch.utils.data import DataLoader

from nintorch.models import construct_model_cifar
from nintorch.utils.perform import disable_debug, enable_tf32, seed_torch, set_benchmark
from utils import get_data_conf, get_datasets, get_transforms, test_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification scripts')
    group = parser.add_argument_group('project')
    group.add_argument('--project-name', type=str, default='nintorch')
    group.add_argument('--run-name', type=str, default='nintorch')
    group.add_argument('--model-log-freq', type=int, default=None)
    group.add_argument('--log-interval', type=int, default=float('inf'))
    group.add_argument('--eval-every-epoch', type=int, default=1)
    group.add_argument('--dist', action='store_true')
    group.add_argument('--yaml-dir', type=str, default=None)
    group.add_argument('--load-dir', type=str, default='./best.pt')

    group = parser.add_argument_group('training')
    group.add_argument('--model-name', type=str, default='resnet20')
    group.add_argument('--lr', type=float, default=1e-3)
    group.add_argument('--opt', type=str, default='adamw')
    group.add_argument('--batch-size', type=int, default=128)
    group.add_argument('--seed', type=int, default=None)
    group.add_argument('--workers', type=int, default=min(os.cpu_count(), 8))
    group.add_argument('--dataset', type=str, default='cifar10')
    group.add_argument('--compile', action='store_true')

    group = parser.add_argument_group('imagenet')
    group.add_argument('--dataset-dir', type=str, default='~/datasets/imagenet')
    args = parser.parse_args()
    args = AttrDict(vars(args))
    filter_warn()

    start = time.perf_counter()
    yaml_log = ''
    if args.yaml_dir is not None:
        args = load_yaml(args.yaml_dir)
        args = AttrDict(args)
        yaml_log = f'Detect `args.yaml` is not None, use arguments in {args.yaml}'

        exp_dir = args.exp_dir
        if exp_dir is None:
            exp_dir = os.path.join(
                'exps',
                str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-'),
            )
        os.makedirs(exp_dir, exist_ok=True)
        set_logger(os.path.join(exp_dir, 'info.log'), stdout=True)
        print(yaml_log)
        print(pformat(args))

    if args.seed is not None:
        seed_torch(args.seed, verbose=True)
    # https://discuss.ray.io/t/amp-mixed-precision-training-is-slower-than-default-precision/9842/7
    enable_tf32(verbose=True) if not args.half else None
    disable_debug(verbose=True)
    set_benchmark(verbose=True)

    data_conf = get_data_conf(args.dataset)
    test_transforms = get_transforms(args, data_conf, only_test=True)
    test_dataset = get_datasets(data_conf, None, test_transforms, only_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (
        data_conf.dataset_name == 'cifar10'
        or data_conf.dataset_name == 'cifar100'
        or data_conf.dataset_name == 'cinic10'
    ):
        model = construct_model_cifar(args.model_name, num_classes=data_conf.num_classes)
        print(f'Construct a `{args.model_name}` model for CIFAR-like dataset.')
    else:
        model = getattr(torchvision.models, args.model_name)(pretrained=False, num_classes=data_conf.num_classes)
        print(f'Construct a `{args.model_name}` model from `torchvision.models`')

    model = model.to(device)
    if args.compile:
        model = torch.compile(model)

    state_dict = torch.load(args.load_dir)
    model_state_dict = state_dict['model_state_dict']
    model.load_state_dict(model_state_dict, strict=False)

    conf = AttrDict(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        test_loader=test_loader,
        model=model,
        best_acc=0.0,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        dist=args.dist,
        test_criterion=nn.CrossEntropyLoss(reduction='mean'),
        eval=True,
    )
    args.update(conf)
    conf = args
    test_epoch(conf)

    end = time.perf_counter()
    runtime = second_to_ddhhmmss(end - start)
    print(f'Total run-time: {runtime} seconds.')
    del test_loader
