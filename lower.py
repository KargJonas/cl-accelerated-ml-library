

# here we take a schedule and find all the necessary buffers
# and kernels that need to be created in order to execute it.

from __future__ import annotations
from typing import Dict, List

from shape import get_nelem
from tensor import Tensor
from ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MetaOps, Op
from dtype import DType, type_sizes
from buffer import Buffer, Base, View


# Defines one input/output of a kernel, it's essentially an interpretation of a buffer
class KernelIO:
    def __init__(self, uid: int, shape: List[int], dtype: DType):
        self.uid, self.shape, self.dtype = uid, shape, dtype

# Kernel descriptors contain all information necessary to generate a kernel.
# These descriptors act as a sort of IR
class Kernel:
    uid: int = 0
    
    def __init__(self, op: Op, in_tensors: List[Tensor], out_tensor: Tensor):
        self.op = op
        # extracts a buffer descriptor from the input/output tensors.
        # such a descriptor consists of the buffer id, shape and datatype
        # todo: when we change the offset of a view, we will need to regenerate the kernel
        # todo: this needs to be fixed when we have more details on kernel generation
        self.inputs = [KernelIO(tensor.buffer.uid, tensor.shape, tensor.dtype) for tensor in in_tensors]
        self.output = KernelIO(out_tensor.buffer.uid, out_tensor.shape, out_tensor.dtype)
        self.uid = Kernel.uid
        self.code: str = None # todo
        Kernel.uid += 1

# todo: remove side-effects
def find_buffers(schedule: List[Tensor]):
    buffers: Dict[int, Buffer] = {}
    
    # we'll just naively create buffers for Unary, Binary, Ternary, Reduce
    # this can later be optimized by fusing kernels and reusing buffers
    for tensor in schedule:
        size = get_nelem(tensor.shape) * type_sizes.get(tensor.dtype)
        if tensor.op in (*UnaryOps, *BinaryOps, *TernaryOps, *ReduceOps, MetaOps.CONST): buffers[tensor.uid] = Base(size)
        elif tensor.op is MetaOps.VIEW:
            # todo: this will cause issues because the parent's buffers have not yet been found at this point in time
            # print(tensor.parents[0].uid)
            # print(buffers[tensor.parents[0].uid])
            buffers[tensor.uid] = View(size, tensor.buffer.get_global_offset(), buffers[tensor.parents[0].uid])
            
        else: raise ValueError(f"Unsupported operation: {tensor.op}")
        # buffers[tensor.buffer.uid] = tensor.buffer # todo: currently buffers are stored in the graph and in this list, which is unnecessary duplication

    return buffers

# todo: remove side-effects
def find_kernels(schedule: List[Tensor]):
    kernels: Dict[int, Kernel] = {}
    
    for tensor in schedule:
        if tensor.op in (MetaOps.CONST, MetaOps.VIEW): continue
        kernel = Kernel(tensor.op, tensor.parents, tensor)
        kernels[kernel.uid] = kernel
