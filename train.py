# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
import hiera
import hiera.train
import hiera.train.config as config

import lightning as L
import torch
import argparse

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import time

def train(model_name: str, config_name: str, train_args: dict = {}):
    start = time.time()
    torch.set_float32_matmul_precision('medium')

    args = getattr(config.TrainArgs, config_name)(model_name).mutate(train_args)

    if "mae" in config_name:
        model = getattr(hiera, f"mae_{model_name}")(pretrained=False, model_name=f"mae_{model_name}")
        engine = hiera.train.MAEEngine(model, args)
        
    else:
        model = getattr(hiera, model_name)(
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
        engine = hiera.train.SupervisedEngine(model, args)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    wandb_logger = WandbLogger(project=f'Hiera_{config_name}', save_dir=args.log_path)

    ckpt_callback = ModelCheckpoint(
        filename='epoch-{epoch}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
        dirpath=args.log_path,
        save_weights_only=True,
    )
    profiler = L.pytorch.profilers.AdvancedProfiler(dirpath=args.log_path)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=max(args.lr_batch_size // args.batch_size, 1),
        accelerator="gpu",

        devices=args.num_gpus,
        num_nodes=args.num_machines,
        strategy="ddp" if args.num_gpus * args.num_machines > 1 else "auto",

        default_root_dir=args.log_path,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
        profiler=profiler,
    )

    trainer.fit(engine, ckpt_path=args.resume if args.resume != "" else None)
    trainer.test(engine)




def make_arg_parser():
    parser = argparse.ArgumentParser(description="Train a Hiera Model")

    parser.add_argument("--model", required=True, type=str, help="Name of the model, e.g. 'hiera_tiny_224'. See hiera.py for available models.")
    parser.add_argument("--config", required=True, type=str, help="Name of the config, e.g. 'in1k_finetune'. See hiera/train/config.py (TrainArgs) for available configs.")
    parser.add_argument('--log-wandb', default=False, action='store_true', help='Log to wandb')
    config.TrainArgs.parse(parser, "train.")

    return parser


def main(args: argparse.Namespace):
    train_args = { k[len("train."):]: v for k, v in vars(args).items() if k.startswith("train.") and v is not None }
    train(args.model, args.config, train_args)

if __name__ == "__main__":
    args = make_arg_parser().parse_args()
    main(args)
