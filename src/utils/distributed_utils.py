"""
Distributed computing utilities for GraphMind
"""

import torch
import numpy as np
from typing import List, Any, Optional, Union
from mpi4py import MPI


def get_world_size() -> int:
    """Get total number of processes in distributed training"""
    return MPI.COMM_WORLD.Get_size()


def get_rank() -> int:
    """Get rank of current process"""
    return MPI.COMM_WORLD.Get_rank()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)"""
    return get_rank() == 0


def all_reduce(
    tensor: torch.Tensor,
    op: str = 'sum',
    comm: Optional[MPI.Comm] = None
) -> torch.Tensor:
    """
    All-reduce operation for distributed training
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'mean', 'max', 'min')
        comm: MPI communicator
        
    Returns:
        Reduced tensor
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    # Convert to numpy for MPI
    np_tensor = tensor.numpy()
    reduced = np.zeros_like(np_tensor)
    
    # Perform reduction
    if op == 'sum':
        comm.Allreduce(np_tensor, reduced, op=MPI.SUM)
    elif op == 'mean':
        comm.Allreduce(np_tensor, reduced, op=MPI.SUM)
        reduced /= comm.Get_size()
    elif op == 'max':
        comm.Allreduce(np_tensor, reduced, op=MPI.MAX)
    elif op == 'min':
        comm.Allreduce(np_tensor, reduced, op=MPI.MIN)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")
    
    return torch.from_numpy(reduced)


def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    comm: Optional[MPI.Comm] = None
) -> torch.Tensor:
    """
    Broadcast tensor from source to all processes
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        comm: MPI communicator
        
    Returns:
        Broadcasted tensor
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    np_tensor = tensor.numpy()
    comm.Bcast(np_tensor, root=src)
    
    return torch.from_numpy(np_tensor)


def gather(
    tensor: torch.Tensor,
    dst: int = 0,
    comm: Optional[MPI.Comm] = None
) -> Optional[List[torch.Tensor]]:
    """
    Gather tensors from all processes to destination
    
    Args:
        tensor: Tensor to gather
        dst: Destination rank
        comm: MPI communicator
        
    Returns:
        List of tensors (only on destination rank)
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Gather tensor shapes first
    shape = np.array(tensor.shape)
    shapes = comm.gather(shape, root=dst)
    
    # Gather tensors
    np_tensor = tensor.numpy()
    gathered = comm.gather(np_tensor, root=dst)
    
    if rank == dst:
        return [torch.from_numpy(g) for g in gathered]
    else:
        return None


def all_gather(
    tensor: torch.Tensor,
    comm: Optional[MPI.Comm] = None
) -> List[torch.Tensor]:
    """
    All-gather operation - gather tensors from all processes to all processes
    
    Args:
        tensor: Tensor to gather
        comm: MPI communicator
        
    Returns:
        List of tensors from all processes
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    np_tensor = tensor.numpy()
    gathered = comm.allgather(np_tensor)
    
    return [torch.from_numpy(g) for g in gathered]


def scatter(
    tensors: Optional[List[torch.Tensor]],
    src: int = 0,
    comm: Optional[MPI.Comm] = None
) -> torch.Tensor:
    """
    Scatter tensors from source to all processes
    
    Args:
        tensors: List of tensors to scatter (only needed on source)
        src: Source rank
        comm: MPI communicator
        
    Returns:
        Scattered tensor for this process
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    
    if rank == src:
        np_tensors = [t.numpy() for t in tensors]
    else:
        np_tensors = None
    
    np_tensor = comm.scatter(np_tensors, root=src)
    
    return torch.from_numpy(np_tensor)


def barrier(comm: Optional[MPI.Comm] = None):
    """
    Synchronization barrier
    
    Args:
        comm: MPI communicator
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    comm.Barrier()


def get_local_rank() -> int:
    """
    Get local rank for multi-GPU nodes
    
    Returns:
        Local rank within the node
    """
    # This is a simplified version
    # In practice, would use SLURM_LOCALID or similar
    return get_rank() % torch.cuda.device_count() if torch.cuda.is_available() else 0


def setup_distributed_training(backend: str = 'nccl'):
    """
    Setup distributed training environment
    
    Args:
        backend: PyTorch distributed backend
    """
    if torch.cuda.is_available():
        # Set CUDA device
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=get_world_size(),
            rank=get_rank()
        )
    else:
        # CPU-only training
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='env://',
            world_size=get_world_size(),
            rank=get_rank()
        )


def cleanup_distributed_training():
    """Cleanup distributed training environment"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class DistributedDataParallel:
    """
    Wrapper for distributed data parallel training
    """
    
    def __init__(self, model: torch.nn.Module, device_ids: Optional[List[int]] = None):
        self.model = model
        self.device_ids = device_ids or [get_local_rank()]
        
        if torch.distributed.is_initialized():
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0]
            )
        else:
            self.ddp_model = model
    
    def forward(self, *args, **kwargs):
        return self.ddp_model(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ddp_model, name)