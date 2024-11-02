from tensor import Tensor
from to_string import schedule_str, tree_str


a = Tensor(1, 3, name="a")
b = Tensor(3, 1, name="b")
d = a.add(b)

d.infer_shape()

print(tree_str(d))
print(schedule_str(d.schedule()))

d.add_reshapes()

print(tree_str(d))
print(schedule_str(d.schedule()))

d.compute_strides()

print(tree_str(d))