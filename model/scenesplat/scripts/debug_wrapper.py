"""
Debug wrapper for multi-node training
"""
import os
import sys
import faulthandler
import traceback

# Enable core dumps
faulthandler.enable()

try:
    # Store original command-line arguments
    original_args = sys.argv[1:]
    
    # Print diagnostic information
    rank = os.environ.get("SLURM_PROCID", "unknown")
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    local_rank = os.environ.get("SLURM_LOCALID", "unknown")
    
    print(f"[DEBUG] Node: {node}, Rank: {rank}, Local Rank: {local_rank}")
    print(f"[DEBUG] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"[DEBUG] Command: {' '.join(original_args)}")
    
    # Import and run the original script
    from tools.train import main
    main()
    
except FloatingPointError as e:
    print(f"[ERROR] Node: {node}, Rank: {rank}: Floating point error: {e}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Node: {node}, Rank: {rank}: Exception: {e}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    sys.exit(1)