from __future__ import annotations
from typing import List
from ops import *

# # this will cause issues with the ternary ops
# def broadcast(a: List[int], b: List[int])-> List[int]:
#     if a is None or b is None: raise ValueError("cannot infer shape from parents")
#     if len(b) > len(a): a, b = b, a     # ensure a is longer or equal to b
#     b = [None] * (len(a) - len(b)) + b  # extend b with None to the left
#     shape = []
#     for (size_a, size_b) in zip(a, b):
#         if size_b == None or size_b == 1 or size_a == size_b: shape.append(size_a)
#         elif size_a == 1: shape.append(size_b)
#         else: raise ValueError(f"shapes incompatible for broadcasting: {a} ~ {b}")
#     return shape

# generalized broadcasting for an arbitrary number of shapes
def broadcast(*shapes: List[List[int]])-> List[int]:
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

class Tensor:
    def __init__(self, *shape, **kwargs):
        self.shape = shape or None
        self.op = kwargs.get("op", MetaOps.CONST)
        self.parents = kwargs.get("parents", None)
        
    def find_shape(self):
        if self.op is MetaOps.CONST: return
        self.shape = broadcast()
        
    def binary(self, op: Op):
        return lambda other: Tensor(op=op, parents=[self, other ])
    
    # If the dimensions of the parents match exactly, we will create only one node that does pairwise add.
    # If the dimensions are broadcastable, we'll introduce the necessary reshapes before the add.
    # Any overhead can later be optimized and fused away.
    # If the dimension are not broadcastable, we'll raise an error.
    def add(self, other: Tensor):
        return Tensor(op=BinaryOps.ADD, parents=[self, other])
        
        
# t = Tensor(3, 2, 4)

print(broadcast([3], [1, 3], [3, 1]))
