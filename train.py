# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import modeling
import modeling.train
import modeling.train.config as config

import lightning as L
import torch
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler

def train(model_name: str,
          config_name: str,
          log_wandb: bool,
          strategy: str,
          torch_compile: bool,
          train_args: dict = {}):
    args = getattr(config.TrainArgs, config_name)(model_name).mutate(train_args)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    save_whole_path = os.path.join(args.log_path, 'save_whole')
    if not os.path.exists(save_whole_path):
        os.makedirs(save_whole_path, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    if "emae" in config_name:
        model = getattr(modeling, f"emae_{model_name}")(pretrained=False, model_name=f"mae_{model_name}")
        engine = modeling.train.EMAEEngine(model, args)
    elif "mae" in config_name:
        model = getattr(modeling, f"mae_{model_name}")(pretrained=False, model_name=f"mae_{model_name}")
        engine = modeling.train.MAEEngine(model, args)
    else:
        model = getattr(modeling, model_name)(
            # Temporary, replace with loading from a checkpoint
            pretrained=False, 
            custom_ckpt_path=args.custom_ckpt_path, 
            custom_ckpt_name=args.custom_ckpt_name, 
            checkpoint=args.custom_ckpt_name,
            num_classes=args.dataset.num_classes,
            model_name=model_name,
            mlp_dropout=args.mlp_dropout,
            expert_dropout=args.expert_dropout,
        )
        engine = modeling.train.SupervisedEngine(model, args)

    if torch_compile:
        engine = torch.compile(engine, mode="reduce-overhead")

    ckpt_callback = ModelCheckpoint(
        filename='epoch-{epoch}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
        dirpath=args.log_path,
        save_weights_only=True,
    )
    ckpt_callback_save_whole = ModelCheckpoint(
        filename='save_all-{epoch}',
        save_top_k=1,
        save_last=True,
        dirpath=save_whole_path,
        save_weights_only=False,
    )
    if log_wandb:
        logger = WandbLogger(project=f'Hiera_{config_name}', save_dir=args.log_path)
    else:
        logger = False

    #profiler = SimpleProfiler(filename=f'{model_name}')

    trainer = L.Trainer(
        # max_steps=1,
        # limit_train_batches=0.01,
        # limit_val_batches=0.1,
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=max(args.lr_batch_size // args.batch_size, 1),
        accelerator="gpu",
        check_val_every_n_epoch=10,

        devices=args.num_gpus,
        num_nodes=args.num_machines,
        strategy=strategy if args.num_gpus * args.num_machines > 1 else "auto",

        default_root_dir=args.log_path,
        logger=logger,
        callbacks=[ckpt_callback, ckpt_callback_save_whole],
        benchmark=True,
        #profiler=profiler
    )

    torch.cuda.empty_cache()
    trainer.fit(engine, ckpt_path=args.resume if args.resume != "" else None)
    trainer.test(engine)




def make_arg_parser():
    parser = argparse.ArgumentParser(description="Train a Hiera Model")

    parser.add_argument("--model", required=True, type=str, help="Name of the model, e.g. 'hiera_tiny_224'. See hiera.py for available models.")
    parser.add_argument("--config", required=True, type=str, help="Name of the config, e.g. 'in1k_finetune'. See hiera/train/config.py (TrainArgs) for available configs.")
    parser.add_argument('--log-wandb', default=False, action='store_true', help='Log to wandb')
    parser.add_argument('--strategy', default='ddp', type=str, choices=['auto', 'ddp', 'fsdp'], help='Training strategy to use. Options: auto, fsdp_native')
    parser.add_argument('--torch_compile', default=False, action='store_true', help='Compile the model with torch.jit.script')
    config.TrainArgs.parse(parser, "train.")

    return parser


def main(args: argparse.Namespace):
    train_args = { k[len("train."):]: v for k, v in vars(args).items() if k.startswith("train.") and v is not None }
    train(args.model, args.config, args.log_wandb, args.strategy, args.torch_compile, train_args)

if __name__ == "__main__":
    args = make_arg_parser().parse_args()
    main(args)
