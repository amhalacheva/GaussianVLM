import os
import torch
import torch.distributed as dist
import socket

def main():
    # Print environment info
    print(f"Hostname: {socket.gethostname()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Environment variables:")
    for var in ['SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS', 'MASTER_ADDR', 'MASTER_PORT']:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Initialize process group
    try:
        rank = int(os.environ.get('SLURM_PROCID', '0'))
        world_size = int(os.environ.get('SLURM_NTASKS', '1'))
        local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
        
        # Set device
        device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        
        print(f"Rank {rank}: Before init_process_group")
        dist.init_process_group(
            backend="NCCL",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank}: After init_process_group")
        
        # Simple collective operation to test
        tensor = torch.ones(1).cuda() * rank
        dist.all_reduce(tensor)
        print(f"Rank {rank}: Tensor sum after all_reduce: {tensor.item()}")
        
        # Clean up
        dist.destroy_process_group()
        print(f"Rank {rank}: Successfully completed")
    
    except Exception as e:
        print(f"Error in process {os.environ.get('SLURM_PROCID', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()