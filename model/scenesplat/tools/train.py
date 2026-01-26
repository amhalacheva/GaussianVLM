"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import torch
import torch.distributed as dist
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.utils import comm  # Import comm module to set _LOCAL_PROCESS_GROUP

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    # Check if running in Slurm environment
    if "SLURM_PROCID" in os.environ and not args.no_distri:
        # When running in Slurm, initialize directly without using the launcher
        # Get Slurm variables
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        gpus_per_node = torch.cuda.device_count()
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        print(f"Rank {rank}: Initializing process group directly from Slurm")
        # Initialize process group
        dist.init_process_group(
            backend="NCCL",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank}: Process group initialized")
        
        # Set up local process group (critical for comm.get_local_rank() to work)
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
        
        # Create local process groups (one per node)
        assert comm._LOCAL_PROCESS_GROUP is None, "Local process group is already created!"
        for i in range(num_nodes):
            # Calculate ranks on this node
            ranks_on_node = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
            pg = dist.new_group(ranks_on_node)
            if i == node_id:
                comm._LOCAL_PROCESS_GROUP = pg
                print(f"Rank {rank}: Created local process group for node {i}: {ranks_on_node}")
        
        # Call main_worker directly
        main_worker(cfg)
    else:
        # Use the launcher for non-Slurm environments
        launch(
            main_worker,
            num_gpus_per_machine=args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            cfg=(cfg,),
        )


if __name__ == "__main__":
    main()