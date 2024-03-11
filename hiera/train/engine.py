# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


from typing import Tuple

from torch.optim.optimizer import Optimizer
from .. import MaskedAutoencoderHiera, Hiera
from . import config
from .utils import make_param_groups, patch_lr_scheduler

from timm.utils import accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def reinit(cls, model: nn.Module, **kwdargs):
    """
    Transfer the given model to our own representation so we can change things like droppath.
    """
    new_model = cls(config = { **model.config, **kwdargs }) # Note assumes model configs are shallow
    new_model.load_state_dict(model.state_dict())
    return new_model



class SupervisedEngine(L.LightningModule):
    
    def __init__(self, model: Hiera, args: config.TrainArgs, **kwargs):
        super().__init__(**kwargs)

        self.args = args
        self.model = reinit(Hiera, model, drop_path_rate=args.drop_path)

        self.save_hyperparameters({"model": model.config, "args": args.save()})

        self.train_loader, self.val_loader = args.make_dataloaders(model.input_size[-1])
        self.mixup = args.dataset.augmentations.mixup(args.label_smoothing, model.num_classes)

        # Extra modules
        if self.mixup is not None:
            self.criterion = SoftTargetCrossEntropy()  # Smoothing is handled by mixup
        elif args.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self) -> torch.Any:
        return self.train_loader
    
    def val_dataloader(self) -> torch.Any:
        return self.val_loader
    
    def test_dataloader(self) -> torch.Any:
        return self.val_loader

    def log_info(self, name, logits, y, loss):
        logits = logits.detach()
        if self.mixup is not None and name == "train":
            # From slowfast, get accuracy of mixup by pooling together the two mixed up classes
            _, top_max_k_inds = torch.topk(y, 2, dim=1, largest=True, sorted=True)
            idx_top1 = torch.arange(y.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(y.shape[0]), top_max_k_inds[:, 1]
            logits[idx_top1] += logits[idx_top2]
            logits[idx_top2] = 0.0
            y = top_max_k_inds[:, 0]

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_acc1", acc1)
        self.log(f"{name}_acc5", acc5)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        if self.mixup is not None:
            x, y = self.mixup(x, y)
        
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.log("lr", self.trainer.optimizers[0].param_groups[-1]["lr"])
        self.log_info("train", logits, y, loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        probs = self.model(x)
        loss = F.nll_loss(torch.log(probs), y)

        self.log_info("val", probs, y, loss)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        probs = self.model(x)
        loss = F.nll_loss(torch.log(probs), y)

        self.log_info("test", probs, y, loss)
    
    def configure_optimizers(self):
        effective_batch_size = self.args.batch_size * self.trainer.accumulate_grad_batches
        scaled_lr = self.args.lr * effective_batch_size / self.args.lr_batch_size

        # Create explicit parameter groups to implement things like layer decay and no weight decay
        params = make_param_groups(self.model, self.args.weight_decay, self.args.layer_decay)

        # To add your own parameters, add a new group, e.g.,
        # params.append({
        #     "params": [p for p in self.my_downstream_model.parameters() if p.requires_grad],
        #     "weight_decay": self.args.weight_decay,
        #     "lr_scale": 1.0,
        # })

        optimizer = self.args.optimizer(
            params,
            lr = scaled_lr,
            weight_decay = self.args.weight_decay
        )

        batches_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        scheduler = self.args.make_lr_scheduler(optimizer, batches_per_epoch)
        
        # Important to apply layer decay (applies lr_scale to each parameter group)
        patch_lr_scheduler(scheduler)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    






class MAEEngine(L.LightningModule):
    
    def __init__(self, model: MaskedAutoencoderHiera, args: config.TrainArgs, **kwargs):
        super().__init__(**kwargs)

        self.args = args
        self.model = reinit(MaskedAutoencoderHiera, model, drop_path_rate=args.drop_path)
        self.save_hyperparameters({"model": model.config, "args": args.save()})

        self.train_loader, self.val_loader = args.make_dataloaders(model.input_size[-1])

    def train_dataloader(self) -> torch.Any:
        return self.train_loader
    
    def val_dataloader(self) -> torch.Any:
        return self.val_loader
    
    def test_dataloader(self) -> torch.Any:
        return self.val_loader

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        loss, _, _, _ = self.model(x)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, _ = batch
        loss, _, _, _ = self.model(x)
        self.log("val_loss", loss)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, _ = batch
        loss, _, _, _ = self.model(x)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        effective_batch_size = self.args.batch_size * self.trainer.accumulate_grad_batches
        scaled_lr = self.args.lr * effective_batch_size / self.args.lr_batch_size

        # Create explicit parameter groups to implement things like layer decay and no weight decay
        params = make_param_groups(self.model, self.args.weight_decay, self.args.layer_decay)

        # To add your own parameters, add a new group, e.g.,
        # params.append({
        #     "params": [p for p in self.my_downstream_model.parameters() if p.requires_grad],
        #     "weight_decay": self.args.weight_decay,
        #     "lr_scale": 1.0,
        # })

        optimizer = self.args.optimizer(
            params,
            lr = scaled_lr,
            weight_decay = self.args.weight_decay
        )

        batches_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        scheduler = self.args.make_lr_scheduler(optimizer, batches_per_epoch)

        # Important to apply layer decay (applies lr_scale to each parameter group)
        patch_lr_scheduler(scheduler)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

