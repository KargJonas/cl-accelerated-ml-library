from lower import find_buffers, find_kernels
from tensor import Tensor
from to_string import schedule_str, tree_str


a = Tensor(1, 3, name="a")
b = Tensor(3, 1, name="b")
d = a.add(b)

d.infer_shape()
d.add_reshapes()
d.compute_strides()

sched = d.schedule()

print(tree_str(d))
print(schedule_str(sched))

buffers = find_buffers(sched)
kernels = find_kernels(sched)

print(buffers[4].buffer)
