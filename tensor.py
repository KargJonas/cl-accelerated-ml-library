from __future__ import annotations
from typing import List
from ops import *

# generalized broadcasting for an arbitrary number of shapes
def broadcast(shapes: List[List[int]])-> List[int]:
    if None in shapes: raise ValueError("cannot infer shape from parents")
    longest = len(max(shapes, key=len))
    shapes = [[1] * (longest - len(shape)) + shape for shape in shapes]
    shape = []
    for sizes in zip(*shapes):
        unique_sizes = set(sizes)
        if len(unique_sizes) == 1: 
            shape.append(sizes[0])
            continue
        non_singleton = [value for value in unique_sizes if value != 1]
        if len(non_singleton) == 1: shape.append(non_singleton[0])
        else: raise ValueError(f"shapes incompatible for broadcasting {'~'.join(shapes)}")
    return shape

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
        self.needs_reshape = False
        
    def find_shape(self):
        if self.op is MetaOps.CONST: return
        elif self.op in UnaryOps: self.shape = self.parents[0].shape
        elif self.op in BinaryOps or self.op in TernaryOps: self.shape = broadcast([parent.shape for parent in self.parents])
        else: raise NotImplementedError(f"Operation \"{self.op}\" not implemented.")
        # todo: for reduce ops, we need meta info
    
    # this is a naive algorithm. later, we should use something smarter to enable multi-gpu computation
    def schedule(self, order: List[Tensor] = []) -> List[Tensor]:
        if self in order and self.op != MetaOps.CONST: raise RecursionError("Detected a cycle in the computation graph. Computation graphs must be acyclic.")
        order = [self] + order
        for parent in self.parents:
            order = parent.schedule() + order
        return order

    # def augment_graph():
    #     ...

    def realize(self):
        sched = self.schedule() # find execution order
        for node in sched: node.find_shape() # use schedule to perform shape inference/propagation
        # at this point we have enough information to generate the kernels and run them

    # If the dimensions of the parents match exactly, we will create only one node that does pairwise add.
    # If the dimensions are broadcastable, we'll introduce the necessary reshapes before the add.
    # Any overhead can later be optimized and fused away.
    # If the dimension are not broadcastable, we'll raise an error.
    def add(self, other: Tensor, **kwargs):
        return Tensor(op=BinaryOps.ADD, parents=[self, other], name=kwargs.get("name"))
