from tensor import Tensor, add_reshapes
from to_string import schedule_str, tree_str


a = Tensor(1, 3, name="a")
b = Tensor(3, 1, name="b")
d = a.add(b)

sched = d.schedule()
for node in sched: node.find_shape()

print(tree_str(d))
print(schedule_str(sched))

add_reshapes(sched)

print(tree_str(d))
print(schedule_str(sched))
