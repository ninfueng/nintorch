import logging
import os

import torch
import torchvision.transforms as T
import wandb
from nincore import AttrDict
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from timm.utils import accuracy
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from nintorch.datasets import CINIC10
from nintorch.utils import AvgMeter

logger = logging.getLogger(__name__)


def get_data_conf(dataset_name: str) -> AttrDict:
    """Get a default configuration for a dataset.

    Args:
        dataset_name: one of `cifar10`, `cifar100`, `cinic10`, and `imagenet`.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        data_conf = AttrDict(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            input_size=(3, 32, 32),
            num_classes=10,
            dataset_name=dataset_name,
        )
    elif dataset_name == 'cifar100':
        data_conf = AttrDict(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
            input_size=(3, 32, 32),
            num_classes=100,
            dataset_name=dataset_name,
        )
    elif dataset_name == 'cinic10':
        data_conf = AttrDict(
            mean=(0.47889522, 0.47227842, 0.43047404),
            std=(0.24205776, 0.23828046, 0.25874835),
            input_size=(3, 32, 32),
            num_classes=10,
            dataset_name=dataset_name,
        )
    elif dataset_name == 'imagenet':
        data_conf = AttrDict(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            input_size=(3, 224, 224),
            num_classes=1_000,
            dataset_name=dataset_name,
        )
    else:
        raise NotImplementedError(
            'Support only `cifar10`, `cifar100`, `cinic10`, and `imagenet`, '
            f'Your: `{dataset_name}`.'
        )
    return data_conf


def get_transforms(
    conf: AttrDict, data_conf: AttrDict, only_test: bool = False
) -> T.Compose | tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(data_conf.mean, data_conf.std)
    use_cifar = False
    if (
        data_conf.dataset_name == 'cifar10'
        or data_conf.dataset_name == 'cifar100'
        or data_conf.dataset_name == 'cinic10'
    ):
        # For `cifar10` size of data, using with `RandomCrop` instead of
        # `RandomResizedCrop`.
        test_transforms = T.Compose([T.ToTensor(), normalize])
        use_cifar = True
    elif data_conf.dataset_name == 'imagenet':
        test_transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError(
            f'Support only `cifar10`, `cifar100`, `cinic10`, and `imagenet`, Your: `{data_conf.data_name}`'
        )

    if only_test:
        return test_transforms

    train_transforms = create_transform(
        input_size=data_conf.input_size[1],
        is_training=True,
        color_jitter=conf.color_jitter if conf.color_jitter else None,
        auto_augment=conf.auto_augment_type if conf.auto_augment else None,
        re_prob=conf.re_prob if conf.rand_erasing else 0.0,
        re_mode=conf.re_mode,
        re_count=conf.re_count,
        interpolation=conf.interpolation,
        mean=data_conf.mean,
        std=data_conf.std,
    )
    if use_cifar:
        train_transforms.transforms[0] = T.RandomCrop(32, padding=4)

    return (train_transforms, test_transforms)


def get_datasets(
    data_conf: AttrDict,
    train_transforms: T.Compose | None,
    test_transforms: T.Compose,
    dataset_dir: str | None = None,
    only_test: bool = False,
) -> Dataset | tuple[Dataset, Dataset]:
    if dataset_dir is not None:
        dataset_dir = os.path.expanduser(dataset_dir)
    else:
        dataset_dir = './datasets'

    if data_conf.dataset_name == 'cifar10':
        test_dataset = CIFAR10(
            root=dataset_dir,
            train=False,
            download=True,
            transform=test_transforms,
        )
        if only_test:
            return test_dataset
        train_dataset = CIFAR10(
            root=dataset_dir,
            train=True,
            download=True,
            transform=train_transforms,
        )

    elif data_conf.dataset_name == 'cifar100':
        test_dataset = CIFAR100(
            root=dataset_dir,
            train=False,
            download=True,
            transform=test_transforms,
        )
        if only_test:
            return test_dataset
        train_dataset = CIFAR100(
            root=dataset_dir,
            train=True,
            download=True,
            transform=train_transforms,
        )

    elif data_conf.dataset_name == 'cinic10':
        dataset_dir = os.path.join(dataset_dir, 'cinic10')
        test_dataset = CINIC10(
            root=dataset_dir, split='test', transforms=test_transforms
        )
        if only_test:
            return test_dataset
        train_dataset = CINIC10(
            root=dataset_dir, split='train', transforms=train_transforms
        )

    elif data_conf.dataset_name == 'imagenet':
        dataset_dir = os.path.join(dataset_dir, 'imagenet')
        test_dataset = ImageFolder(os.path.join(dataset_dir, 'val'), test_transforms)
        if only_test:
            return test_dataset
        train_dataset = ImageFolder(
            os.path.join(dataset_dir, 'train'), train_transforms
        )

    else:
        raise NotImplementedError(
            f'`dataset` only supports `cifar10`, `cifar100`, `cinic10`, and `imagenet`, Your: `{data_conf.dataset}`'
        )
    return (train_dataset, test_dataset)


def load_model_optim_sche(
    load_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: bool = False,
) -> tuple[float, int]:
    """Load a model, optimizer, and scheduler from a .pt with `load_dir` directory."""

    load_dir = os.path.expanduser(load_dir)
    state_dict = torch.load(load_dir)

    if ema:
        model_state_dict = state_dict['model_ema_state_dict']
    else:
        model_state_dict = state_dict['model_state_dict']
    optimizer_state_dict = state_dict['optimizer_state_dict']
    scheduler_state_dict = state_dict['scheduler_state_dict']

    model.load_state_dict(model_state_dict, strict=False)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)

    accuracy = state_dict['accuracy']
    start_epoch = state_dict['epoch']
    return (accuracy, start_epoch)


def train_epoch(conf: AttrDict) -> None:
    conf.model.train()
    train_len = len(conf.train_loader)
    top1, top5, losses = AvgMeter(), AvgMeter(), AvgMeter()

    for batch_idx, (inputs, targets) in enumerate(conf.train_loader):
        inputs = inputs.to(
            conf.device,
            non_blocking=True,
            memory_format=torch.channels_last if conf.chl_last else None,
        )
        targets = targets.to(conf.device, non_blocking=True)
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        conf.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=conf.half, dtype=torch.bfloat16):
            if conf.mixup and conf.mixup_fn is not None:
                # If using `timm`, it expects to pass `timm.data.mixup.Mixup` as `conf.mixup_fn`.
                mixup_inputs, mixup_targets = conf.mixup_fn(inputs, targets)
                outputs = conf.model(mixup_inputs)
                loss = conf.criterion(outputs, mixup_targets)
            else:
                outputs = conf.model(inputs)
                loss = conf.criterion(outputs, targets)

            # Always divided by a constant but `loss.backward()` will accumulate gradients without
            # updating from `optimizer.step()`.
            if conf.grad_accum is not None:
                loss /= conf.grad_accum

        if conf.half:
            loss = conf.scaler.scale(loss)
            loss.backward()

            if conf.clip_grad > 0.0:
                conf.scaler.unscale_(conf.optimizer)
                torch.nn.utils.clip_grad_norm_(conf.model.parameters(), conf.clip_grad)

            if conf.grad_accum is None or (batch_idx + 1) % conf.grad_accum == 0:
                conf.scaler.step(conf.optimizer)
                conf.scaler.update()

        else:
            loss.backward()
            if conf.clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(conf.model.parameters(), conf.clip_grad)

            # Guard for second statement % by 0.
            if conf.grad_accum is None or (batch_idx + 1) % conf.grad_accum == 0:
                conf.optimizer.step()

        if conf.model_ema is not None:
            conf.model_ema.update(conf.model)

        with torch.inference_mode():
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            batch_size = targets.size(0)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            losses.update(loss.item(), batch_size)

        if conf.dist and batch_idx == train_len - 1:
            top1.all_reduce()
            top5.all_reduce()
            losses.all_reduce()

        # If `conf.rank == 0`, allows only every `conf.log_freq` or on the last iterations.
        if (
            (batch_idx + 1) % conf.log_freq == 0
            or batch_idx == train_len - 1
            and conf.rank == 0
        ):
            first_param = conf.optimizer.param_groups[0]
            cur_lr = first_param['lr']
            msg = (
                f'Tr E {conf.epoch_idx} ({batch_idx + 1}/{train_len}) | '
                f'L: {losses.avg / (batch_idx + 1):.3e} | '
                f'A1: {top1.avg:.2f} ({int(top1.sum / 100.)}/{top1.count}) | '
                f'Lr: {cur_lr:.3e} |'
            )
            logging.info(msg)

            if conf.wandb and batch_idx == train_len - 1:
                wandb.log(
                    {
                        'train_loss': losses.avg,
                        'train_acc': top1.avg,
                        'train_acc5': top5.avg,
                    },
                    step=conf.epoch_idx,
                )

        if conf.warmup_scheduler is not None and not conf.warmup_scheduler.done:
            conf.warmup_scheduler.step()

    # If `conf.scheduler` is not None, start only when `conf.warmup_scheduler` is done or None.
    if (
        conf.warmup_scheduler is None
        or conf.warmup_scheduler.done
        and (conf.scheduler is not None)
    ):
        conf.scheduler.step()


@torch.inference_mode()
def test_epoch(conf: AttrDict) -> None:
    conf.model.eval()
    test_len = len(conf.test_loader)
    top1, top5, losses = AvgMeter(), AvgMeter(), AvgMeter()

    for batch_idx, (inputs, targets) in enumerate(conf.test_loader):
        inputs = inputs.to(
            conf.device,
            non_blocking=True,
            memory_format=torch.channels_last if conf.chl_last else None,
        )
        targets = targets.to(conf.device, non_blocking=True)
        if conf.model_ema is None:
            outputs = conf.model(inputs)
        else:
            outputs = conf.model_ema.module(inputs)
        loss = conf.test_criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, (1, 5))
        batch_size = targets.size(0)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        losses.update(loss.item(), batch_size)

        if conf.dist and batch_idx == test_len - 1:
            top1.all_reduce()
            top5.all_reduce()
            losses.all_reduce()

        if (
            (batch_idx + 1) % conf.log_freq == 0
            or batch_idx == test_len - 1
            and conf.rank == 0
        ):
            msg = (
                f'Te E {conf.epoch_idx:,} ({batch_idx + 1:,}/{test_len:,}) | '
                f'L: {losses.avg / (batch_idx + 1):.3e} | '
                f'A1: {top1.avg:.2f} ({int(top1.sum / 100.):,}/{top1.count:,}) | '
                f'A5: {top5.avg:.2f} ({int(top5.sum / 100.):,}/{top5.count:,}) | '
            )

            if batch_idx == test_len - 1:
                if conf.wandb:
                    wandb.log(
                        {
                            'test_loss': losses.avg,
                            'test_acc': top1.avg,
                            'test_acc5': top5.avg,
                        },
                        step=conf.epoch_idx,
                    )
                if top1.avg > conf.best_acc:
                    best_acc = top1.avg
                    if conf.save_model:
                        # https://github.com/pytorch/pytorch/issues/9176#issuecomment-403570715
                        try:
                            model_state_dict = conf.model.module.state_dict()
                        except AttributeError:
                            model_state_dict = conf.model.state_dict()
                        model_ema_state_dict = None

                        if conf.model_ema is not None:
                            model_ema_state_dict = conf.model_ema.module.state_dict()

                        # Cannot use with `AttrDict`. Must use with `dict` to save.
                        state = AttrDict(
                            model_state_dict=model_state_dict,
                            model_ema_state_dict=model_ema_state_dict,
                            optimizer_state_dict=(
                                conf.optimizer.state_dict()
                                if conf.optimizer is not None
                                else None
                            ),
                            scheduler_state_dict=(
                                conf.scheduler.state_dict()
                                if conf.scheduler is not None
                                else None
                            ),
                            accuracy=best_acc,
                            epoch=conf.epoch_idx,
                            seed=conf.seed,
                            rng_state=torch.get_rng_state(),
                        )
                        if conf.half:
                            state.update(scaler_state_dict=conf.scaler.state_dict())

                        save_model_dir = os.path.join(conf.exp_dir, 'best.pt')
                        torch.save(state, save_model_dir)
                        msg += 'S'
                    conf.best_acc = best_acc
            logging.info(msg)

    if conf.print_eval:
        msg = (
            f'Te E {conf.epoch_idx} ({batch_idx + 1}/{test_len}) | '
            f'L: {losses.avg / (batch_idx + 1):.3e} | '
            f'A1: {top1.avg:.2f} ({int(top1.sum / 100.)}/{top1.count}) | '
            f'A5: {top5.avg:.2f} ({int(top5.sum / 100.)}/{top5.count}) | '
        )
        print(msg)
