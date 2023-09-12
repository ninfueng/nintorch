import logging
import os

import torch
import wandb
from nincore import AttrDict
from torch.cuda.amp import autocast

from nintorch.utils import AvgMeter
from timm.utils import accuracy

logger = logging.getLogger(__name__)


try:
    inference_mode = torch.inference_mode
except AttributeError:
    inference_mode = torch.no_grad


def train_an_epoch(conf: AttrDict) -> None:
    conf.model.train()
    train_len = len(conf.train_loader)
    top1, top5, losses = AvgMeter(), AvgMeter(), AvgMeter()

    for batch_idx, (inputs, targets) in enumerate(conf.train_loader):
        inputs = inputs.to(conf.device, non_blocking=True)
        targets = targets.to(conf.device, non_blocking=True)
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        conf.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=conf.fp16, dtype=torch.bfloat16):
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

        if conf.fp16:
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

        with inference_mode():
            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            batch_size = targets.size(0)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            losses.update(loss.item(), batch_size)

        if conf.dist and batch_idx == train_len - 1:
            top1.all_reduce()
            top5.all_reduce()
            losses.all_reduce()

        # If `conf.rank == 0`, allows only every `conf.log_interval` or on the last iterations.
        if (
            (batch_idx + 1) % conf.log_interval == 0
            or batch_idx == train_len - 1
            and conf.rank == 0
        ):
            first_param = conf.optimizer.param_groups[0]
            cur_lr = first_param["lr"]
            cur_weight_decay = first_param["weight_decay"]
            msg = (
                f"Train Epoch {conf.epoch_idx} ({batch_idx + 1}/{train_len}) | "
                f"Loss: {losses.avg / (batch_idx + 1):.3e} | "
                f"Acc: {top1.avg:.2f} ({int(top1.sum / 100.)}/{top1.count}) | "
                f"Decay: {cur_weight_decay:.3e} | "
                f"Lr: {cur_lr:.3e} |"
            )
            logging.info(msg)

            if conf.wandb and batch_idx == train_len - 1:
                wandb.log(
                    {"train_loss": losses.avg, "train_acc": top1.avg},
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


@inference_mode()
def test_an_epoch(conf: AttrDict) -> None:
    conf.model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_len = len(conf.test_loader)
    top1, top5, losses = AvgMeter(), AvgMeter(), AvgMeter()

    for batch_idx, (inputs, targets) in enumerate(conf.test_loader):
        inputs = inputs.to(conf.device, non_blocking=True)
        targets = targets.to(conf.device, non_blocking=True)
        outputs = conf.model(inputs)
        loss = criterion(outputs, targets)

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
            (batch_idx + 1) % conf.log_interval == 0
            or batch_idx == test_len - 1
            and conf.rank == 0
        ):
            msg = (
                f"Test  Epoch {conf.epoch_idx} ({batch_idx + 1}/{test_len}) | "
                f"Loss: {losses.avg / (batch_idx + 1):.3e} | "
                f"Acc: {top1.avg:.2f} ({int(top1.sum / 100.)}/{top1.count}) | "
            )
            logging.info(msg)

            if batch_idx == test_len - 1:
                if conf.wandb:
                    wandb.log(
                        {"test_loss": losses.avg, "test_acc": top1.avg},
                        step=conf.epoch_idx,
                    )

                if top1.avg > conf.best_acc:
                    # https://github.com/pytorch/pytorch/issues/9176#issuecomment-403570715
                    try:
                        model_state_dict = conf.model.module.state_dict()
                    except AttributeError:
                        model_state_dict = conf.model.state_dict()

                    best_acc = top1.avg
                    # Cannot use with `AttrDict`. Must use with `dict` to save.
                    state = dict(
                        model_state_dict=model_state_dict,
                        optimizer_state_dict=conf.optimizer.state_dict(),
                        scheduler_state_dict=conf.scheduler.state_dict(),
                        accuracy=best_acc,
                        epoch=conf.epoch_idx,
                        seed=conf.seed,
                        rng_state=torch.get_rng_state(),
                    )

                    if conf.fp16:
                        state.update(scaler_state_dict=conf.scaler.state_dict())

                    save_dir = os.path.join(conf.exp_path, "checkpoint")
                    os.makedirs(save_dir, exist_ok=True)
                    save_model_dir = os.path.join(save_dir, "best.pth")
                    torch.save(state, save_model_dir)
                    logger.info(
                        f"Saving a model with Test Acc@{conf.epoch_idx}: {top1.avg:.4f}"
                    )
                    conf.best_acc = best_acc
