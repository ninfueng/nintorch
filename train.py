"""Train CIFAR10, CIFAR100, CINIC10, and ImageNet with PyTorch.

Example:
>>> python main.py

Example for distributed training:

>>> python -m torch.distributed.launch --nnodes=1 --node_rank=0\
           --master_addr='127.0.0.1' --master_port=1234\
           --nproc_per_node=2 --env  main.py

Note:
* One node per computer.
* One node per a process and multi-processes.
* One process per a GPU.
* world_size = num_nodes * num_gpus_per_node
* global_rank = num_gpus_per_node * node_idx + gpu_idx (or local_rank)
* batch_per_rank = batch / num_gpus_per_node
* worker_per_rank = total_worker_per_node / num_gpus_per_node

All local rank = 0 will save, log, and accumulate all in its node.
"""
import argparse
import datetime
import logging
import os
import sys
import time
from functools import reduce
from pprint import pformat

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models
import wandb
from nincore import AttrDict
from nincore.io import load_yaml
from nincore.time import second_to_ddhhmmss
from nincore.utils import backup_scripts, filter_warn, set_logger
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim.optim_factory import create_optimizer_v2
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import BatchSampler, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from nintorch.models import construct_model_cifar
from nintorch.scheduler import WarmupLR
from nintorch.utils.perform import disable_debug, enable_tf32, seed_torch, set_benchmark
from utils import get_data_conf, get_datasets, get_transforms, load_model_optim_sche, test_epoch, train_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification scripts')
    group = parser.add_argument_group('project')
    group.add_argument('--project-name', type=str, default='nintorch')
    group.add_argument('--run-name', type=str, default='nintorch')
    group.add_argument('--model-log-freq', type=int, default=None)
    group.add_argument('--log-freq', type=int, default=float('inf'))
    group.add_argument('--eval-every-epoch', type=int, default=1)
    group.add_argument('--dist', action='store_true')
    group.add_argument('--yaml-dir', type=str, default=None)

    group = parser.add_argument_group('training')
    group.add_argument('--model-name', type=str, default='resnet20')
    group.add_argument('--lr', type=float, default=1e-3)
    group.add_argument('--opt', type=str, default='adamw')
    group.add_argument('--momentum', type=float, default=0.9)
    group.add_argument('--batch-size', type=int, default=128)
    group.add_argument('--seed', type=int, default=None)
    group.add_argument('--workers', type=int, default=min(os.cpu_count(), 8))
    group.add_argument('--weight-decay', type=float, default=0.0)
    group.add_argument('--warmup-epoch', type=int, default=5)
    group.add_argument('--epoch', type=int, default=200)
    group.add_argument('--clip-grad', type=float, default=0.0)
    group.add_argument('--dataset', type=str, default='cifar10')
    group.add_argument('--subset-idx-load-dir', type=str, default=None)
    group.add_argument('--grad-accum', type=int, default=None)
    group.add_argument('--exp-dir', type=str, default=None)
    group.add_argument('--label-smoothing', action='store_true')
    group.add_argument('--wandb', action='store_true')
    group.add_argument('--half', action='store_true')
    group.add_argument('--compile', action='store_true')
    group.add_argument('--chl-last', action='store_true')
    group.add_argument('--load-dir', type=str, default=None)

    group = parser.add_argument_group('mixup')
    group.add_argument('--mixup', action='store_true')
    group.add_argument('--mixup-mixup-alpha', type=float, default=0.8)
    group.add_argument('--mixup-cutmix-alpha', type=float, default=1.0)
    group.add_argument('--mixup-cutmix-minmax', type=float, default=None)
    group.add_argument('--mixup-prob', type=float, default=1.0)
    group.add_argument('--mixup-switch-prob', type=float, default=0.5)
    group.add_argument('--mixup-mode', type=str, default='batch')
    group.add_argument('--mixup-label-smoothing', type=float, default=0.1)

    group = parser.add_argument_group('augment')
    group.add_argument('--auto-augment', action='store_true')
    group.add_argument('--rand-erasing', action='store_true')
    group.add_argument('--color-jitter', type=float, default=0.4)
    group.add_argument('--auto-augment-type', type=str, default='rand-m9-mstd0.5-inc1')
    group.add_argument('--interpolation', type=str, default='bilinear')

    group = parser.add_argument_group('erasing')
    group.add_argument('--re-prob', type=float, default=0.25)
    group.add_argument('--re-mode', type=str, default='pixel')
    group.add_argument('--re-count', type=float, default=1)

    group = parser.add_argument_group('imagenet')
    group.add_argument('--dataset-dir', type=str, default='~/datasets/imagenet')
    args = parser.parse_args()
    args = AttrDict(vars(args))

    log_rank_zero = lambda info: logging.info(info) if rank == 0 else None
    filter_warn()
    start = time.perf_counter()

    yaml_log = ''
    if args.yaml_dir is not None:
        args = load_yaml(args.yaml_dir)
        args = AttrDict(args)
        yaml_log = f'Detect `args.yaml` is not None, use arguments in {args.yaml}'

    if args.dist:
        master_addr = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_idx = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(gpu_idx)
        batch_size = args.batch_size // world_size
        num_workers = args.workers // world_size

        dist_info = f"""\n
            World {world_size}, Rank: {rank}, GPU: {gpu_idx},
            Batch size: {batch_size}, Number worker: {num_workers},
            Master address: port: {master_addr}:{master_port}\n"""
        print(dist_info)

        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
    else:
        rank = gpu_idx = 0
        batch_size = args.batch_size

    if rank == 0:
        if args.wandb:
            wandb.login()
            wandb.init(project=args.project_name, config=vars(args), allow_val_change=True)
            wandb.run.name = args.run_name

        exp_dir = args.exp_dir
        if exp_dir is None:
            exp_dir = os.path.join(
                'exps',
                str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-'),
            )
        os.makedirs(exp_dir, exist_ok=True)
        set_logger(os.path.join(exp_dir, 'info.log'), stdout=True)
        log_rank_zero(yaml_log)
        log_rank_zero(pformat(args))

        backup_scripts(['*.py'], os.path.join(exp_dir, 'scripts'))
        cmd = 'python ' + reduce(lambda x, y: f'{x} {y}', sys.argv)

        log_rank_zero(f'Run with the command line: `{cmd}`.')
    else:
        exp_dir = None

    if args.seed is not None:
        seed_torch(args.seed, verbose=True)
    # https://discuss.ray.io/t/amp-mixed-precision-training-is-slower-than-default-precision/9842/7
    enable_tf32(verbose=True) if not args.half else None
    disable_debug(verbose=True)
    set_benchmark(verbose=True)

    data_conf = get_data_conf(args.dataset)
    train_transforms, test_transforms = get_transforms(args, data_conf)
    train_dataset, test_dataset = get_datasets(data_conf, train_transforms, test_transforms, args.dataset_dir)
    log_rank_zero(train_transforms)
    log_rank_zero(test_transforms)

    if args.mixup is True:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_mixup_alpha,
            cutmix_alpha=args.mixup_cutmix_alpha,
            cutmix_minmax=args.mixup_cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.mixup_label_smoothing if args.label_smoothing else 0.0,
            num_classes=data_conf.num_classes,
        )
        log_rank_zero('Using `mixup` data augmentation.')
    else:
        mixup_fn = None
        log_rank_zero('Not using `mixup` data augmentation.')

    if args.subset_idx_load_dir is not None:
        log_rank_zero(f'Detect `args.subset_idx_load_dir is not None`. Use subset of training dataset.')
        subset_idx = torch.load(args.subset_idx_load_dir)
        train_dataset = Subset(train_dataset, subset_idx)
        log_rank_zero(f'Using subset with training data shape: `{subset_idx.shape}`.')

    if args.dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)
    else:
        train_sampler = test_sampler = train_batch_sampler = None

    train_loader = DataLoader(
        train_dataset,
        # `batch_size = 1` is a default value, which will be overwritten by sampler.
        batch_size=batch_size if not args.dist else 1,
        shuffle=True if not args.dist else None,
        batch_sampler=train_batch_sampler if args.dist else None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False if not args.dist else None,
        sampler=test_sampler if args.dist else None,
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
        log_rank_zero(f'Construct a `{args.model_name}` model for CIFAR-like dataset.')
    else:
        model = getattr(torchvision.models, args.model_name)(pretrained=False, num_classes=data_conf.num_classes)
        log_rank_zero(f'Construct a `{args.model_name}` model from `torchvision.models`')

    from nineff.low.ternary import TerConv2d, TerLinear

    from bin_aware import BinAwareCrossEntropy
    from bin_quant import BinConv2d, BinLinear
    from nintorch.utils import convert_layer
    from reparam import ReparamConv2d, ReparamLinear
    from swin_mlp import SwinMLP

    # convert_layer(model, nn.Conv2d, TerConv2d)
    # convert_layer(model, nn.Linear, TerLinear)
    # convert_layer(model, nn.Conv2d, BinConv2d)
    # convert_layer(model, nn.Linear, BinLinear)
    # convert_layer(model, nn.Conv2d, ReparamConv2d)
    # convert_layer(model, nn.Linear, ReparamLinear)
    # criterion = BinAwareCrossEntropy(model)
    model = SwinMLP()
    print(model)
    model = model.to(device, non_blocking=True, memory_format=torch.channels_last if args.chl_last else None)
    if args.compile:
        model = torch.compile(model)
        log_rank_zero(f'Use `torch.compile` with default settings.')
    log_rank_zero(model)

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
        log_rank_zero(
            f'Detect mixup, using `SoftTargetCrossEntropy` with label smoothing: {args.mixup_label_smoothing}'
        )
    elif args.label_smoothing and args.mixup_label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.mixup_label_smoothing)
        log_rank_zero(
            'Did not detect `mixup` but `mixup_label_smoothing` > 0.0, '
            f'using `LabelSmoothingCrossEntropy` with {args.mixup_label_smoothing}.'
        )
    else:
        criterion = nn.CrossEntropyLoss()
        log_rank_zero(
            f'Did not detect `mixup`, `label_smoothing`, or `mixup_label_smoothing` == 0.0, '
            'using `CrossEntropyLoss`.'
        )
    test_criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        filter_bias_and_bn=True,
    )
    log_rank_zero(optimizer)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    log_rank_zero(f'Scheduler state-dict: \n {scheduler.state_dict()}')

    if args.warmup_epoch > 0:
        warmup_scheduler = WarmupLR(
            optimizer,
            max_iterations=args.warmup_epoch * len(train_loader),
            max_lr=args.lr,
        )
        args.epoch += args.warmup_epoch
        log_rank_zero(f'Using warmup learning rate for adamw`{args.warmup_epoch}` epochs.')
    else:
        warmup_scheduler = None

    scaler = opt_level = None
    if args.half:
        scaler = GradScaler()
        log_rank_zero('Using `torch.cuda.amp` half.')

        if args.dist:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(gpu_idx)
            model = DDP(model, device_ids=(gpu_idx,))
            log_rank_zero('Convert `BatchNorm` to `SyncBatchNorm` and using `torch DistributedDataParallel`.')

    if not args.dist:
        model = nn.DataParallel(model)
        log_rank_zero('Wrap model with `nn.DataParallel`.')

    if args.compile:
        model = torch.compile(model)
        log_rank_zero(f'Use `torch.compile` with default settings.')

    if args.load_dir is not None:
        _model = model
        if hasattr(model, 'module'):
            _model = model.module
        acc, start_epoch = load_model_optim_sche(args.load_dir, _model, optimizer, scheduler)
        log_rank_zero(f'Load a model with Acc: {acc}, Epoch: {start_epoch} from `{args.load_dir}`.')

    conf = AttrDict(
        device=gpu_idx if args.dist else device,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        test_criterion=test_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        scaler=scaler,
        opt_level=opt_level,
        mixup_fn=mixup_fn,
        best_acc=0.0,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        rank=rank,
        dist=args.dist,
        exp_dir=exp_dir,
    )
    args.update(conf)
    conf = args

    if conf.rank == 0 and conf.wandb and conf.model_log_freq is not None:
        wandb.watch(model, log_freq=conf.model_log_freq)
        log_rank_zero('Using `wandb` to tracking the model distribution.')

    for epoch_idx in range(1, args.epoch + 1):
        conf.epoch_idx = epoch_idx
        train_epoch(conf)
        if epoch_idx % conf.eval_every_epoch == 0:
            test_epoch(conf)

    end = time.perf_counter()

    if conf.rank == 0:
        runtime = second_to_ddhhmmss(end - start)
        log_rank_zero(f'Total run-time: {runtime} seconds.')
        conf.runtime = runtime
        conf.to_json(os.path.join(conf.exp_dir, 'settings.json'))

        if args.wandb:
            wandb.run.summary['best_acc'] = conf.best_acc
            wandb.config.update(conf, allow_val_change=True)
            wandb.finish()
        print(conf.best_acc)

    if args.dist:
        dist.destroy_process_group()

    del train_loader
    del test_loader
