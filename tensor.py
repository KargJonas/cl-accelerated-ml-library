from __future__ import annotations
from typing import List
from ops import *

from shape import broadcast, get_contiguous_strides, get_nelem
from dtype import DType, type_sizes
from buffer import Buffer

# this will be the order of operations:
#   1. create graph
#   2. find schedule
#   3. propagate shapes
#   4. patch reshapes into the graph where necessary.
#      (e.g Tensor(3) + Tensor(3, 1) -> Tensor(3, 1) will be converted
#      to (Tensor(3).reshape(3, 1)) + Tensor(3, 1) -> Tensor(3, 1))
#      since reshapes don't hold data and don't need execution,
#      it's not necessary recompute the schedule (todo: validate)
#   5. use now available shapes along with the ops of each tensor
#      to generate kernel and buffer specifications
#   6. use specs to generate opencl code

class Tensor:
    id_counter = 0

    def __init__(self, *shape, **kwargs):
        self.uid = Tensor.id_counter
        Tensor.id_counter += 1
        self.shape: List[int] = list(shape) or None
        self.op: Op = kwargs.get("op", MetaOps.CONST)
        self.parents: List[Tensor] = kwargs.get("parents", [])
        self.name: str = kwargs.get("name", None)
        self.dtype: str = kwargs.get("dtype", DType.F16)
        self.buffer: Buffer = None # later, we will either assign a view or base buffer to this property
        self.strides: List[int] = None
        if self.dtype not in type_sizes: raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.op is MetaOps.CONST and not self.shape: raise ValueError("Cannot create a const without shape.")
    
    # infers the shape of the current node using the parent nodes' shapes (recursively)
    def infer_shape(self):
        if self.op is MetaOps.CONST: return
            
        for parent in self.parents: parent.infer_shape()
        
        if self.op in UnaryOps: self.shape = self.parents[0].shape
        elif self.op in BinaryOps or self.op in TernaryOps: self.shape = broadcast([parent.shape for parent in self.parents])
        elif self.op is MetaOps.VIEW: return
        else: raise NotImplementedError(f"Operation \"{self.op}\" not implemented.")
        # todo: for reduce ops, we need meta info
    
    def compute_strides(self):
        # Recursive anchor
        # The recursion will stop when we encounter nodes that don't have parents (i.e., const nodes)
        for parent in self.parents:
            parent.compute_strides()
        
        if self.op is MetaOps.VIEW:
            # todo: check if this will also work with views of views
            new_strides = []
            orig_shape = self.parents[0].shape
            orig_strides = self.parents[0].strides
            for i in range(len(self.shape)):
                orig_dim = len(orig_shape) - (len(self.shape) - i)
                if orig_dim < 0: new_strides.append(0)
                else: new_strides.append(0 if orig_shape[orig_dim] == 1 else orig_strides[orig_dim])
            self.strides = new_strides
        else: self.strides = get_contiguous_strides(self.shape)

    # inserts reshapes where necessary for broadcasting
    # e.g. (showing shape, not data):
    #   [3] + [3, 1] -> (reshape [3] to [3, 3]) + (reshape [3, 1] to [3, 3])
    def add_reshapes(self):
        for i, parent in enumerate(self.parents):
            if parent.shape != self.shape and self.op != MetaOps.VIEW:
                self.parents[i] = Tensor(*self.shape, op=MetaOps.VIEW, parents=[parent])
            parent.add_reshapes()
        
    # this is a naive algorithm. later, we should use something smarter to enable multi-gpu computation
    def schedule(self, order: List[Tensor] = []) -> List[Tensor]:
        if self in order and self.op != MetaOps.CONST:
            raise RecursionError("Detected a cycle in the computation graph. Computation graphs must be acyclic.")
        
        order = [self] + order
        for parent in self.parents:
            order = parent.schedule() + order
        
        return order

    def realize(self):
        self.infer_shape()
        self.add_reshapes()
        sched = self.schedule()
        # todo: generate kernels and buffers for schedule, run schedule

    # If the dimensions of the parents match exactly, we will create only one node that does pairwise add.
    # If the dimensions are broadcastable, we'll introduce the necessary reshapes before the add.
    # Any overhead can later be optimized and fused away.
    # If the dimension are not broadcastable, we'll raise an error.
    def add(self, other: Tensor, **kwargs) -> Tensor:
        return Tensor(op=BinaryOps.ADD, parents=[self, other], name=kwargs.get("name"))

    def log2(self, **kwargs) -> Tensor:
        return Tensor(op=UnaryOps.LOG2, parents=[self], name=kwargs.get("name"))

    def view(self, *shape, **kwargs):
        # Note: internally these types of broadcasts are allowed for broadcasting purposes
        #       but there, we adjust the strides such that we duplicate elements along some axis
        if get_nelem(shape) != get_nelem(self.shape):
            raise ValueError(f"Cannot create view {shape} from tensor of shape {self.shape}.")
        return Tensor(*shape, op=MetaOps.VIEW, parents=[self], name=kwargs.get("name"))


