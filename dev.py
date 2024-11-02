from tensor import Tensor
from to_string import schedule_str, tree_str


a = Tensor(3, 3, name="a")
b = Tensor(3, 1, name="b")
c = a.add(b, name="c")
d = c.add(a, name="d")

sched = d.schedule()

print(tree_str(d))
print(schedule_str(sched))

for node in sched:
    node.find_shape()
    print(node.name, node.shape)
