# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# Adapted from facebookresearch/mae
# --------------------------------------------------------

import argparse
import os
import uuid

import examples.train as train
import submitit
from pathlib import Path


def parse_args():
    train_parser = train.make_arg_parser()
    parser = argparse.ArgumentParser("Submitit for training a Hiera model", parents=[train_parser])
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import examples.train as train
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args["train.log_path"] = Path(str(self.args["train.log_path"]).replace("%j", str(job_env.job_id)))
        
        train.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        checkpoint_folder = os.path.join(self.args["train.log_path"], "lightning_logs", "version_0", "checkpoints")
        if os.path.exists(checkpoint_folder):
            checkpoint_file = None
            for file in os.listdir(checkpoint_folder):
                if file.endswith(".ckpt"):
                    checkpoint_file = os.path.join(checkpoint_folder, file)
                    break
            
            if checkpoint_file is not None:
                self.args["train.resume"] = checkpoint_file
        
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)


def main():
    args = parse_args()
    if args["train.log_path"] is None:
        args["train.log_path"] = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args["train.log_path"], slurm_max_num_timeout=30)

    if args["train.num_machines"] is None:
        args["train.num_machines"] = 1
    if args["train.num_gpus"] is None:
        args["train.num_gpus"] = 1

    num_gpus_per_node = args["train.num_gpus"]
    nodes = args["train.num_machines"]
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="hiera")

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()