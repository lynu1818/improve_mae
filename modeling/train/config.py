# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
import tqdm
import typing
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

import timm.data
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .utils import config, field

from flash.core.optimizers import LAMB


@config
class Optimizer:
    # "adam", "adamw", or "sgd"
    type: str

    # only for adam
    beta1: float
    beta2: float

    # only for sgd
    momentum: float


    @classmethod
    def adamw(cls, beta1: float = 0.9, beta2: float = 0.999):
        return cls("adamw", beta1=beta1, beta2=beta2, momentum=0.0)
    
    @classmethod
    def adam(cls, beta1: float = 0.9, beta2: float = 0.999):
        return cls("adam", beta1=beta1, beta2=beta2, momentum=0.0)
    
    @classmethod
    def sgd(cls, momentum: float = 0.9):
        return cls("sgd", momentum=momentum, beta1=0, beta2=0)
    
    @classmethod
    def lamb(cls):
        return cls("lamb", beta1=0.9, beta2=0.999)
    
    def __call__(self, params, lr: float, weight_decay: float):
        if self.type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=lr,
                betas= (self.beta1, self.beta2),
                weight_decay=weight_decay
            )
        elif self.type == "adam":
            return torch.optim.Adam(
                params,
                lr=lr,
                betas= (self.beta1, self.beta2),
                weight_decay=weight_decay
            )
        elif self.type == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=self.momentum, weight_decay=weight_decay)
        elif self.type == 'lamb':
            return LAMB(params, lr=lr, betas=(self.beta1, self.beta2), weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type {self.type}.")


@config
class LRScheduler:
    # "cosine" or "linear"
    type: str
    
    # Start at <ratio_start> * lr, end at <ratio_end> * lr
    ratio_start: float
    ratio_end  : float

    @classmethod
    def cosine(cls, ratio_end: float = 0.0):
        return cls("cosine", ratio_start=1.0, ratio_end=ratio_end)
    
    @classmethod
    def linear(cls, ratio_start: float = 1.0, ratio_end: float = 0.0):
        return cls("linear", ratio_start=ratio_start, ratio_end=ratio_end)
    

    def __call__(self, optimizer: torch.optim.Optimizer, total_iters: int):
        if self.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, total_iters, optimizer.defaults["lr"] * self.ratio_end)
        elif self.type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, max(self.ratio_start, 4e-08), self.ratio_end, total_iters)
        else:
            raise ValueError(f"Unknown LR scheduler type {self.type}.")



@config
class Mixup:
    alpha: float = 0.8        # Mixup alpha, mixup is disabled if alpha is 0.0
    cutmix_alpha: float = 1.0 # Cutmix alpha, cutmix is disabled if alpha is 0.0
    prob: float = 1.0         # Probability of applying a mixup or cutmix operation
    switch_prob: float = 0.5  # Probably of switching between mixup and cutmix when both enabled
    mode: str = "batch"       # How to apply mixup/cutmix params: per 'batch', 'pair', or 'elem'

    def __call__(self, label_smoothing: float, num_classes: int):
        if self.alpha > 0.0 or self.cutmix_alpha > 0.0:
            return timm.data.Mixup(
                mixup_alpha=self.alpha, cutmix_alpha=self.cutmix_alpha,
                prob=self.prob, switch_prob=self.switch_prob, mode=self.mode,
                label_smoothing=label_smoothing, num_classes=num_classes
            )
        else:
            return None

    @classmethod
    def none(cls):
        return cls(alpha=0.0, cutmix_alpha=0.0, prob=0.0)




@config
class Augmentations:

    mean: typing.List[float] = field(default_factory=lambda: IMAGENET_DEFAULT_MEAN)
    std:  typing.List[float] = field(default_factory=lambda: IMAGENET_DEFAULT_STD)

    # Simple augmentations (if auto_augment is "")
    min_scale: float = 0.2
    max_scale: float = 1.0
    horizontal_flip_prob: float = 0.5

    auto_augment: str = "rand-m9-mstd0.5-inc1" # See timm rand_augment_transform

    
    # Random Erase Params
    random_erase_prob: float = 0.25  # Probability of applying a random erase, inherited from MAE
    random_erase_mode: str = "pixel" # Random erase mode: 'pixel' or 'image'
    random_erase_count: int = 1      # Number of random erases to apply

    mixup: Mixup = Mixup()

    def __call__(self, train: bool, img_size: int) -> v2.Compose:
        if train:
            if self.auto_augment != "":
                return create_transform(
                    input_size=img_size,
                    is_training=True,
                    auto_augment=self.auto_augment,
                    mean=self.mean,
                    std=self.std,
                    interpolation='bicubic',
                    re_prob=self.random_erase_prob,
                    re_mode=self.random_erase_mode,
                    re_count=self.random_erase_count,
                    color_jitter=None,
                )
            else:
                # Simple MAE augmentations
                return v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.RandomResizedCrop(
                        img_size,
                        scale=(self.min_scale, self.max_scale),
                        interpolation=v2.InterpolationMode.BICUBIC
                    ),
                    v2.RandomHorizontalFlip(self.horizontal_flip_prob),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std)
                ])
        else:
            crop_pct = 1.0 if img_size > 224 else 224 / 256
            size = int(img_size / crop_pct)

            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(img_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ])
        



    @classmethod
    def mae(cls):
        return cls(
            auto_augment="",
            random_erase_prob=0.0,
            mixup = Mixup.none()
        )
    
    @classmethod
    def supervised(cls):
        return cls() # Default augmentations (auto-augment, etc.)
    
    @classmethod
    def none(cls):
        return cls(
            auto_augment="",
            random_erase_prob=0.0,
            mixup = Mixup.none(),
            min_scale=1.0,
            max_scale=1.0,
            horizontal_flip_prob=0.0,
        )
        


class InMemoryImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # Load all images and labels into memory
        print(f"Loading {len(self.samples)} images into memory...")
        for path, label in tqdm.tqdm(self.samples):
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                self.images.append(img)
                self.labels.append(label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

class SA1B_Dataset(Dataset):
    def __init__(self, root, pct=1.0, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []

        # Get the list of directories
        dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.startswith('sa_')])
        num_dirs = int(1000 * pct)
        dirs = dirs[:num_dirs]

        # Preload all image paths and labels
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.jpg'):
                    file_path = os.path.join(dir_path, file_name)
                    self.image_paths.append(file_path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, -1.

    def __len__(self):
        return len(self.image_paths)

@config
class Dataset:

    path: str = "/mnt/home/andyqmongo/data/imagenet/imagenet"
    type: str = "imagefolder"  # ("imagenet", "imagefolder", or "sa1b"), "imagefolder" should have "train" and "val" subfolders
    augmentations: Augmentations = Augmentations.supervised()
    num_classes: int = 1000
    sa1b_pct: float = 0.1
    sa1b_path = "/mnt/home/andyqmongo/data/SA-1B/SA-1B-Downloader/sa1b_extracted/"

    def __call__(self, train: bool, img_size: int) -> torch.utils.data.Dataset:
        """ Construct a pytorch dataset from the given configuration. """
        transform = self.augmentations(train, img_size)
        print(f'Preparing dataset {self.type}')
        if self.type.find("sa1b") >= 0:
            dataset = SA1B_Dataset(self.sa1b_path, self.sa1b_pct, transform=transform)
        elif self.type == "imagefolder":
            root = os.path.join(self.path, "train" if train else "val")
            dataset = datasets.ImageFolder(root, transform=transform)
            #dataset = InMemoryImageFolder(root, transform=transform)
        elif self.type == "imagenet":
            dataset = datasets.ImageNet(self.path, "train" if train else "val", transform=transform)
        else:
            raise ValueError(f"Unknown dataset type {self.type}.")
        print('Dataset prepared')
        return dataset







@config
class TrainArgs:
    precision: str = "16-mixed"  # Recommended: "16-mixed" or "32"
    log_path: str = "./"         # Directory to save logs+checkpoints to (will save to {log_dir}/)
    resume: str = ""             # Path to a checkpoint to resume training from

    custom_ckpt_path: str = ""         # Path to a checkpoint to load the model from (For fine-tuning)
    custom_ckpt_name: str = ""         # Name of the model in the checkpoint (For fine-tuning)

    # Regularization
    label_smoothing: float = 0.1
    weight_decay: float = 0.05

    # Important regularization parameters
    layer_decay: float = 1.0          # Layer-wise decay: 1.0 means no decay, 0.0 means no learning
    drop_path: float = 0.1            # Probability of dropping a path
    mlp_dropout: float = 0.0          # Dropout rate for MLPs
    expert_dropout: float = 0.0  # [ADDED] Dropout rate for experts. Switch Transformers.

    batch_size: int = 512      # TOTAL batch size across _all_ gpus
    num_workers: int = 16
    prefetch_factor: int = 2

    # Batch size will be automatically split evenly among gpus*machines and lr will be scaled accordingly
    num_machines: int = 1     # Number of machines
    num_gpus: int = 1         # Number of gpus per machine

    lr: float = 8e-4          # The global learning rate (i.e., if batch_size = lr_batch_size)
    lr_batch_size: int = 1024 # The batch size this learning rate was meant for
    
    optimizer: Optimizer = Optimizer.adamw()

    epochs: int = 400
    warmup_epochs: int = 40

    warmup_scheduler: LRScheduler = LRScheduler.linear(0.0, 1.0)
    lr_scheduler: LRScheduler = LRScheduler.cosine(0.0)

    dataset: Dataset = Dataset()

    # For MAE only
    mask_ratio: float = 0.75



    def make_dataloaders(self, img_size: int):
        train_loader = DataLoader(
            self.dataset(train=True, img_size=img_size),
            batch_size=self.batch_size // (self.num_machines * self.num_gpus),
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True, # Important for mixup
        )

        val_loader = DataLoader(
            self.dataset(train=False, img_size=img_size),
            batch_size=self.batch_size // (self.num_machines * self.num_gpus),
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
        return train_loader, val_loader


    def make_lr_scheduler(self, optimizer: torch.optim.Optimizer, batches_per_epoch: int):
        total_warmup_iters = self.warmup_epochs * batches_per_epoch
        rest_iters = self.epochs * batches_per_epoch - total_warmup_iters

        lr_scheduler = self.lr_scheduler(optimizer, rest_iters)

        if total_warmup_iters == 0:
            return lr_scheduler
        else:
            warmup_scheduler = self.warmup_scheduler(optimizer, total_warmup_iters)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [warmup_scheduler, lr_scheduler],
                [total_warmup_iters]
            )



    @classmethod
    def in1k_finetune(cls, model: str):
        args = {
            "hiera_tiny_224":      { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_small_224":     { "lr": 2e-3, "epochs": 200, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_base_224":      { "lr": 2e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_base_plus_224": { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_large_224":     { "lr": 1e-3, "epochs":  50, "drop_path": 0.1, "layer_decay": 0.85 },
            "hiera_huge_224":      { "lr": 1e-3, "epochs":  50, "drop_path": 0.3, "layer_decay": 0.85 },

            "hiera_tiny_224_st_moe_0001":          { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_tiny_224_st_moe_50p":           { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_tiny_224_st_moe_0011_50p":      { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            
            "hiera_base_plus_224_st_moe_0011_50p": { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_large_st_moe_0011_50p":         { "lr": 1e-3, "epochs":  50, "drop_path": 0.1, "layer_decay": 0.85 },


            "hiera_tiny_512":                      { "lr": 2e-3, "epochs": 300, "drop_path": 0.2, "layer_decay": 0.65 },
            "hiera_tiny_512_st_moe_0011_50p":      { "lr": 2e-3, "epochs": 300, "drop_path": 0.2, "layer_decay": 0.65 },
            "hiera_base_plus_512":                 { "lr": 1e-3, "epochs": 100, "drop_path": 0.2, "layer_decay": 0.7  },
            "hiera_base_plus_512_st_moe_0011_50p": { "lr": 1e-3, "epochs": 100, "drop_path": 0.2, "layer_decay": 0.7  },

            "hieradet_tiny_224":                      { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_abs_win_tiny_224":                 { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_abs_win_tiny_224_st_moe_0011_50p": { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_abs_win_tiny_512":                 { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },
            "hiera_abs_win_tiny_512_st_moe_0011_50p": { "lr": 2e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.65 },

            "hiera_abs_win_base_plus_224":                 { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_abs_win_base_plus_448":                 { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_abs_win_base_plus_512":                 { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": { "lr": 1e-3, "epochs": 100, "drop_path": 0.1, "layer_decay": 0.7  },
            
            "vit_base_224": { "lr": 1e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.7 },
            "vit_large_224": { "lr": 1e-3, "epochs": 300, "drop_path": 0.2, "layer_decay": 0.75  },
            "vit_huge_224": { "lr": 1e-3, "epochs": 300, "drop_path": 0.3, "layer_decay": 0.7  },
            "vit_base_448": { "lr": 1e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.7  },
            "vit_large_448": { "lr": 1e-3, "epochs": 300, "drop_path": 0.2, "layer_decay": 0.75  },
            "vit_base_512": { "lr": 1e-3, "epochs": 300, "drop_path": 0.1, "layer_decay": 0.7  },
            "vit_large_512": { "lr": 1e-3, "epochs": 300, "drop_path": 0.2, "layer_decay": 0.75  },
        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for finetuning.")

        return cls(
            batch_size=16,
            label_smoothing=0.1,
            weight_decay=0.05,
            lr_batch_size=1024,
            mask_ratio=0.0,
            warmup_epochs=5,
            
            dataset=Dataset(augmentations = Augmentations.supervised()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.999),

            **args[model]
        )

    @classmethod
    def in1k_mae(cls, model: str):
        args = {
            "hiera_tiny_224":      {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_small_224":     {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_base_224":      {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_base_plus_224": {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_large_224":     {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_huge_224":      {"lr": 8e-4, "drop_path": 0.3 },

            "hiera_tiny_224_st_moe_0001":          {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_50p":           {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_0011_50p":      {"lr": 8e-4, "drop_path": 0.0 },

            "hiera_tiny_512":                      {"lr": 8e-4, "drop_path": 0.1 },
            "hiera_tiny_512_st_moe_0011_50p":      {"lr": 8e-4, "drop_path": 0.1 },
            "hiera_base_plus_512":                 {"lr": 8e-4, "drop_path": 0.3 },
            "hiera_base_plus_512_st_moe_0011_50p": {"lr": 8e-4, "drop_path": 0.3 },

            "hieradet_tiny_224": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_224": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_224_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_512": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_512_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.0},


            "hiera_abs_win_base_plus_224": {"lr": 8e-4, "drop_path": 0.2},
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.2},
            "hiera_abs_win_base_plus_512": {"lr": 8e-4,"drop_path": 0.2},
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.2},

            "vit_base_224": { "lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_224": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_huge_224": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_base_448": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_448": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_base_512": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_512": {"lr": 1.5e-4, "drop_path": 0.0},

        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for MAE training.")

        return cls(
            weight_decay=0.05,
            layer_decay=1.0,
            batch_size=128,
            lr_batch_size=4096,
            warmup_epochs=40,
            mask_ratio=0.6,
            epochs=400,

            dataset=Dataset(augmentations = Augmentations.mae()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.95),
            **args[model]
        )

    @classmethod
    def in1k_mae_plus(cls, model: str):
        args = {
            "hiera_tiny_224":      {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_small_224":     {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_base_224":      {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_base_plus_224": {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_large_224":     {"lr": 8e-4, "drop_path": 0.2 },
            "hiera_huge_224":      {"lr": 8e-4, "drop_path": 0.3 },

            "hiera_tiny_224_st_moe_0001":          {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_50p":           {"lr": 8e-4, "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_0011_50p":      {"lr": 8e-4, "drop_path": 0.0 },

            "hiera_tiny_512":                      {"lr": 8e-4, "drop_path": 0.1 },
            "hiera_tiny_512_st_moe_0011_50p":      {"lr": 8e-4, "drop_path": 0.1 },
            "hiera_base_plus_512":                 {"lr": 8e-4, "drop_path": 0.3 },
            "hiera_base_plus_512_st_moe_0011_50p": {"lr": 8e-4, "drop_path": 0.3 },

            "hieradet_tiny_224": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_224": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_224_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_512": {"lr": 8e-4,"drop_path": 0.0},
            "hiera_abs_win_tiny_512_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.0},


            "hiera_abs_win_base_plus_224": {"lr": 8e-4, "drop_path": 0.2},
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.2},
            "hiera_abs_win_base_plus_512": {"lr": 8e-4,"drop_path": 0.2},
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": {"lr": 8e-4,"drop_path": 0.2},

            "vit_base_224": { "lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_224": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_huge_224": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_base_448": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_448": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_base_512": {"lr": 1.5e-4, "drop_path": 0.0},
            "vit_large_512": {"lr": 1.5e-4, "drop_path": 0.0},

        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for MAE Plus training.")

        return cls(
            weight_decay=0.05,
            layer_decay=1.0,
            batch_size=128,
            lr_batch_size=4096,
            warmup_epochs=40,
            mask_ratio=0.6,
            epochs=400,

            dataset=Dataset(augmentations = Augmentations.mae()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.95),
            **args[model]
        )

    @classmethod
    def in1k_emae(cls, model: str):
        args = {
            "hiera_tiny_224":      { "drop_path": 0.0 },
            "hiera_small_224":     { "drop_path": 0.0 },
            "hiera_base_224":      { "drop_path": 0.2 },
            "hiera_base_plus_224": { "drop_path": 0.2 },
            "hiera_large_224":     { "drop_path": 0.2 },
            "hiera_huge_224":      { "drop_path": 0.3 },

            "hiera_tiny_224_st_moe_0001":          { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_50p":           { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_0011_50p":      { "drop_path": 0.0 },

            "hiera_tiny_512":                      { "drop_path": 0.1 },
            "hiera_tiny_512_st_moe_0011_50p":      { "drop_path": 0.1 },
            "hiera_base_plus_512":                 { "drop_path": 0.3 },
            "hiera_base_plus_512_st_moe_0011_50p": { "drop_path": 0.3 },

            "hieradet_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224_st_moe_0011_50p": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512_st_moe_0011_50p": {"drop_path": 0.0},


            "hiera_abs_win_base_plus_224": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_448": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": {"drop_path": 0.2},

            "vit_base_224": {"drop_path": 0.1},
            "vit_large_224": {"drop_path": 0.1},
            "vit_huge_224": {"drop_path": 0.1},
            "vit_base_448": {"drop_path": 0.1},
            "vit_large_448": {"drop_path": 0.1},
            "vit_base_512": {"drop_path": 0.1},
            "vit_large_512": {"drop_path": 0.1},

        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for MAE training.")

        return cls(
            weight_decay=0.05,
            layer_decay=1.0,
            batch_size=32,
            lr=1.5e-4,
            lr_batch_size=4096,
            warmup_epochs=40,
            mask_ratio=0.75,
            epochs=400,

            dataset=Dataset(augmentations = Augmentations.mae()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.95),
            **args[model]
        )

    @classmethod
    def in1k_emae_plus(cls, model: str):
        args = {
            "hiera_tiny_224":      { "drop_path": 0.0 },
            "hiera_small_224":     { "drop_path": 0.0 },
            "hiera_base_224":      { "drop_path": 0.2 },
            "hiera_base_plus_224": { "drop_path": 0.2 },
            "hiera_large_224":     { "drop_path": 0.2 },
            "hiera_huge_224":      { "drop_path": 0.3 },

            "hiera_tiny_224_st_moe_0001":          { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_50p":           { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_0011_50p":      { "drop_path": 0.0 },

            "hiera_tiny_512":                      { "drop_path": 0.1 },
            "hiera_tiny_512_st_moe_0011_50p":      { "drop_path": 0.1 },
            "hiera_base_plus_512":                 { "drop_path": 0.3 },
            "hiera_base_plus_512_st_moe_0011_50p": { "drop_path": 0.3 },

            "hieradet_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224_st_moe_0011_50p": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512_st_moe_0011_50p": {"drop_path": 0.0},


            "hiera_abs_win_base_plus_224": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_448": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": {"drop_path": 0.2},

            "vit_base_224": {"drop_path": 0.1},
            "vit_large_224": {"drop_path": 0.1},
            "vit_huge_224": {"drop_path": 0.1},
            "vit_base_448": {"drop_path": 0.1},
            "vit_large_448": {"drop_path": 0.1},
            "vit_base_512": {"drop_path": 0.1},
            "vit_large_512": {"drop_path": 0.1},

        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for MAE training.")

        return cls(
            weight_decay=0.05,
            layer_decay=1.0,
            batch_size=32,
            lr=1.5e-4,
            lr_batch_size=4096,
            warmup_epochs=40,
            mask_ratio=0.75,
            epochs=400,

            dataset=Dataset(augmentations = Augmentations.mae()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.95),
            **args[model]
        )

    @classmethod
    def sa1b_mae(cls, model: str):
        args = {
            "hiera_tiny_224":      { "drop_path": 0.0 },
            "hiera_small_224":     { "drop_path": 0.0 },
            "hiera_base_224":      { "drop_path": 0.2 },
            "hiera_base_plus_224": { "drop_path": 0.2 },
            "hiera_large_224":     { "drop_path": 0.2 },
            "hiera_huge_224":      { "drop_path": 0.3 },

            "hiera_tiny_224_st_moe_0001":          { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_50p":           { "drop_path": 0.0 },
            "hiera_tiny_224_st_moe_0011_50p":      { "drop_path": 0.0 },

            "hiera_tiny_512":                      { "drop_path": 0.1 },
            "hiera_tiny_512_st_moe_0011_50p":      { "drop_path": 0.1 },
            "hiera_base_plus_512":                 { "drop_path": 0.3 },
            "hiera_base_plus_512_st_moe_0011_50p": { "drop_path": 0.3 },

            "hieradet_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224": {"drop_path": 0.0},
            "hiera_abs_win_tiny_224_st_moe_0011_50p": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512": {"drop_path": 0.0},
            "hiera_abs_win_tiny_512_st_moe_0011_50p": {"drop_path": 0.0},


            "hiera_abs_win_base_plus_224": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_224_st_moe_0011_50p": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_448": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512": {"drop_path": 0.2},
            "hiera_abs_win_base_plus_512_st_moe_0011_50p": {"drop_path": 0.2},

            "vit_base_224": {"drop_path": 0.1},
            "vit_large_224": {"drop_path": 0.1},
            "vit_huge_224": {"drop_path": 0.1},
            "vit_base_448": {"drop_path": 0.1},
            "vit_base_448": {"drop_path": 0.1},
            "vit_base_512": {"drop_path": 0.1},
            "vit_large_512": {"drop_path": 0.1},

        }

        if model not in args:
            raise ValueError(f"Unknown model {model} for MAE training.")

        return cls(
            weight_decay=0.05,
            layer_decay=1.0,
            batch_size=128,
            lr=8e-4,
            lr_batch_size=4096,
            warmup_epochs=40,
            mask_ratio=0.6,
            epochs=400,

            dataset=Dataset(type='sa1b', augmentations = Augmentations.mae()),
            optimizer=Optimizer.adamw(beta1=0.9, beta2=0.95),
            **args[model]
        )
