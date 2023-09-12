"""Train CIFAR10, CIFAR100, CINIC10, and ImageNet with PyTorch.

Example:
>>> python main.py

Example for distributed training:

>>> python -m torch.distributed.launch --nnodes=1 --node_rank=0\
           --master_addr="127.0.0.1" --master_port=1234\
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
import warnings
from functools import reduce

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
import wandb
from nincore import AttrDict
from nincore.io import load_yaml
from nincore.time import second_to_ddhhmmss
from timm.data import Mixup
from timm.data.transforms_factory import create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim.optim_factory import create_optimizer_v2
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import BatchSampler, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from nintorch.datasets.cinic10 import CINIC10
from nintorch.models import construct_model_cifar
from nintorch.scheduler import WarmupLR
from nintorch.utils import backup_scripts, set_logger
from nintorch.utils.perform import enable_perform, set_random_seed
from train import test_an_epoch, train_an_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification template for CIFAR10, CIFAR100, CINIC10, and ImageNet."
    )
    group = parser.add_argument_group("Project related settings.")
    group.add_argument("--project-name", type=str, default="nintorch")
    group.add_argument("--run-name", type=str, default="nintorch")
    group.add_argument("--model-log-freq", type=int, default=None)
    group.add_argument("--log-interval", type=int, default=100)
    group.add_argument("--eval-every-epoch", type=int, default=1)
    group.add_argument("--record-imgs", action="store_true")
    group.add_argument("--dist", action="store_true")
    group.add_argument("--yaml-dir", type=str, default=None)

    group = parser.add_argument_group("Training related settings.")
    group.add_argument("--model-name", type=str, default="resnet20")
    group.add_argument("--lr", type=float, default=5e-1)
    group.add_argument("--opt", type=str, default="sgd")
    group.add_argument("--batch-size", type=int, default=512)
    group.add_argument("--seed", type=int, default=3_407)
    group.add_argument("--workers", type=int, default=os.cpu_count())
    group.add_argument("--weight-decay", type=float, default=1e-4)
    group.add_argument("--warmup-epoch", type=int, default=5)
    group.add_argument("--epoch", type=int, default=200)
    group.add_argument("--clip-grad", type=float, default=0.0)
    group.add_argument("--dataset", type=str, default="cifar10")
    group.add_argument("--download-dir", type=str, default="./downloads")
    group.add_argument("--subset-idx-load-dir", type=str, default=None)
    group.add_argument("--grad-accum", type=int, default=None)
    group.add_argument("--fp16", action="store_true")
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--mixup", action="store_true")
    group.add_argument("--auto-augment", action="store_true")
    group.add_argument("--rand-erasing", action="store_true")
    group.add_argument("--label-smoothing", action="store_true")
    group.add_argument("--compile", action="store_true")

    group = parser.add_argument_group("Mixup related settings.")
    group.add_argument("--mixup-mixup-alpha", type=float, default=0.8)
    group.add_argument("--mixup-cutmix-alpha", type=float, default=1.0)
    group.add_argument("--mixup-cutmix-minmax", type=float, default=None)
    group.add_argument("--mixup-prob", type=float, default=1.0)
    group.add_argument("--mixup-switch-prob", type=float, default=0.5)
    group.add_argument("--mixup-mode", type=str, default="batch")
    group.add_argument("--mixup-label-smoothing", type=float, default=0.1)

    group = parser.add_argument_group("Augmentation related settings.")
    group.add_argument("--color-jitter", type=float, default=0.4)
    group.add_argument("--auto-augment-type", type=str, default="rand-m9-mstd0.5-inc1")
    group.add_argument("--interpolation", type=str, default="bicubic")
    # Random-erasing related arguments.
    group.add_argument("--re-prob", type=float, default=0.25)
    group.add_argument("--re-mode", type=str, default="pixel")
    group.add_argument("--re-count", type=float, default=1)

    group = parser.add_argument_group("ImageNet related settings.")
    group.add_argument("--dataset-dir", type=str, default="~/datasets/imagenet")
    args = parser.parse_args()

    # `rank` is not defined that is still fine.
    log_rank_zero = lambda info: logging.info(info) if rank == 0 else None
    start = time.perf_counter()
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    args = AttrDict(vars(args))

    # If using `yaml-dir`, all other `args` will be discards and prioritize `args` from `yaml`.
    yaml = False
    if args.yaml_dir is not None:
        args = load_yaml(args.yaml_dir)
        args = AttrDict(args)
        yaml = True

    if args.dist:
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu_idx = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(gpu_idx)
        batch_size = args.batch_size // world_size
        num_workers = args.workers // world_size

        dist_info = f"""\n
            World {world_size}, Rank: {rank}, GPU: {gpu_idx},
            Batch size: {batch_size}, Number worker: {num_workers},
            Master address: port: {master_addr}:{master_port}\n"""
        print(dist_info)

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
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
            wandb.init(
                project=args.project_name, config=vars(args), allow_val_change=True
            )
            wandb.run.name = args.run_name

        exp_path = os.path.join(
            "experiments",
            str(datetime.datetime.now())
            .replace(":", "-")
            .replace(".", "-")
            .replace(" ", "-"),
        )
        os.makedirs(exp_path, exist_ok=True)
        set_logger(os.path.join(exp_path, "info.log"), to_stdout=True, with_color=True)
        logging.info(args)

        if yaml:
            logging.info(
                "Detect `args.yaml_dir` is not None, "
                f"use all arguments based on {args.yaml_dir}."
            )

        backup_scripts(["*.py", "*.sh", "*.md"], os.path.join(exp_path, "scripts"))
        cmd = "python " + reduce(lambda x, y: f"{x} {y}", sys.argv)
        set_random_seed(args.seed)
        enable_perform()
        logging.info(f"Run with the command line: `{cmd}`.")
    else:
        exp_path = None

    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
        img_size = 32

    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        img_size = 32

    elif args.dataset == "cinic10":
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        num_classes = 10
        img_size = 32

    elif args.dataset == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        num_classes = 1_000
        img_size = 224

    else:
        raise NotImplementedError(
            "Support only `cifar10`, `cifar100`, `cinic10`, and `imagenet`, "
            f"but your input: {args.dataset}"
        )

    normalize = transforms.Normalize(mean, std)
    train_transforms = create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=args.color_jitter if args.color_jitter else None,
        auto_augment=args.auto_augment_type if args.auto_augment else None,
        re_prob=args.re_prob if args.random_erasing else 0.0,
        re_mode=args.re_mode,
        re_count=args.recount,
        interpolation=args.interpolation,
    )

    if img_size == 32:
        # For `cifar10` size of data, using with `RandomCrop` instead of `RandomResizedCrop`.
        train_transforms.transforms[0] = transforms.RandomCrop(img_size, padding=4)
        test_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    elif img_size == 224:
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError(
            f"Detected `img_size` not in [32, 224], your `img_size`: {img_size}"
        )
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
            num_classes=num_classes,
        )
        log_rank_zero("Using `mixup` data augmentation.")
    else:
        mixup_fn = None
        log_rank_zero("Not using `mixup` data augmentation.")

    if args.dataset == "cifar10":
        train_dataset = CIFAR10(
            root=args.download_dir,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_dataset = CIFAR10(
            root=args.download_dir,
            train=False,
            download=True,
            transform=test_transforms,
        )
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100(
            root=args.download_dir,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_dataset = CIFAR100(
            root=args.download_dir,
            train=False,
            download=True,
            transform=test_transforms,
        )
    elif args.dataset == "cinic10":
        download_dir = os.path.join(args.download_dir, "cinic10")
        train_dataset = CINIC10(
            root=download_dir, mode="train", transforms=train_transforms
        )
        test_dataset = CINIC10(
            root=download_dir, mode="test", transforms=test_transforms
        )
    elif args.dataset == "imagenet":
        dataset_dir = os.path.expanduser(args.dataset_dir)
        log_rank_zero(f"Loading imagenet dataset from: `{dataset_dir}`")

        train_dataset = ImageFolder(
            os.path.join(dataset_dir, "train"), train_transforms
        )
        test_dataset = ImageFolder(os.path.join(dataset_dir, "val"), test_transforms)
    else:
        raise NotImplementedError(
            "`dataset` only supports `cifar10`, `cifar100`, `cinic10`, and `imagenet` "
            f"Your: `{args.dataset}`"
        )

    if args.subset_idx_load_dir is not None:
        log_rank_zero(
            f"Detect `args.subset_idx_load_dir is not None`."
            "Use subset of training dataset."
        )
        subset_idx = torch.load(args.subset_idx_load_dir)
        train_dataset = Subset(train_dataset, subset_idx)
        log_rank_zero(f"Using subset with training data shape: `{subset_idx.shape}`.")

    if args.dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)
    else:
        train_sampler = test_sampler = train_batch_sampler = None

    train_loader = DataLoader(
        train_dataset,
        # batch_size = 1 is a default value which will be overwritten by sampler.
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if img_size == 32:
        model = construct_model_cifar(args.model_name, num_classes=num_classes)
        log_rank_zero(f"Construct a `{args.model_name}` model for CIFAR-like dataset.")
    else:
        model = getattr(torchvision.models, args.model_name)()
        log_rank_zero(
            f"Construct a `{args.model_name}` model from `torchvision.models`"
        )

    model = model.to(device)
    if args.compile:
        model = torch.compile(model)
    log_rank_zero(model)

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
        log_rank_zero(
            f"Detect mixup, using `SoftTargetCrossEntropy` with"
            f"label smoothing: {args.mixup_label_smoothing}"
        )
    elif args.label_smoothing and args.mixup_label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.mixup_label_smoothing)
        log_rank_zero(
            "Did not detect `mixup` but `mixup_label_smoothing` > 0.0, "
            f"using `LabelSmoothingCrossEntropy` with {args.mixup_label_smoothing}."
        )
    else:
        criterion = nn.CrossEntropyLoss()
        log_rank_zero(
            f"Did not detect `mixup`, `label_smoothing`, or `mixup_label_smoothing` == 0.0, "
            "using `CrossEntropyLoss`."
        )

    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
        filter_bias_and_bn=True,
    )
    log_rank_zero(optimizer)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    log_rank_zero(f"Scheduler state-dict: \n {scheduler.state_dict()}")

    if args.warmup_epoch > 0:
        warmup_scheduler = WarmupLR(
            optimizer,
            max_iterations=args.warmup_epoch * len(train_loader),
            max_lr=args.lr,
        )
        args.epoch += args.warmup_epoch
        log_rank_zero(f"Using warmup learning rate for `{args.warmup_epoch}` epochs.")
    else:
        warmup_scheduler = None

    scaler = opt_level = None
    if args.fp16:
        scaler = GradScaler()
        log_rank_zero("Using `torch.cuda.amp` FP16.")

        if args.dist:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(gpu_idx)
            model = DDP(model, device_ids=(gpu_idx,))
            log_rank_zero(
                "Convert `BatchNorm` to `SyncBatchNorm` and using `torch DistributedDataParallel`."
            )

    if not args.dist:
        model = nn.DataParallel(model)
        log_rank_zero("Wrap model with `nn.DataParallel`.")

    conf = AttrDict(
        device=gpu_idx if args.dist else device,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
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
        exp_path=exp_path,
    )
    conf.update(args)
    if conf.rank == 0 and conf.wandb and conf.model_log_freq is not None:
        wandb.watch(model, log_freq=conf.model_log_freq)
        log_rank_zero("Using `wandb` to tracking the model distribution.")

    start_epoch = 0
    for epoch_idx in range(1, args.epoch + 1):
        conf.epoch_idx = epoch_idx
        train_an_epoch(conf)
        if epoch_idx % conf.eval_every_epoch == 0:
            test_an_epoch(conf)
    end = time.perf_counter()
    del train_loader
    del test_loader

    if conf.rank == 0:
        runtime = second_to_ddhhmmss(end - start)
        logging.info(f"Total run-time: {runtime}.")
        conf.runtime = runtime
        conf.to_json(os.path.join(exp_path, "settings.json"))

        if args.wandb:
            wandb.run.summary["best_acc"] = conf.best_acc
            wandb.config.update(conf, allow_val_change=True)
            wandb.finish()
        print(conf.best_acc)

    if args.dist:
        dist.destroy_process_group()
