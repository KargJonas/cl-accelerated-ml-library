from typing import List
import math

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

# computes the number of elements in a tensor, given the shape
def get_nelem(shape: List[int]) -> int:
    return 0 if not shape else math.prod(shape)

# computes the strides for a contiguous tensor
def get_contiguous_strides(shape: List[int]) -> List[int]:
    strides: List[int] = [1]
    prod = 1
    for i in range(len(shape) - 2, -1, -1):
        prod *= shape[i + 1]
        strides.insert(0, prod)
    return strides
