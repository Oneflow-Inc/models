import oneflow as flow

class GraphModel(flow.nn.Graph):
  def __init__(self) -> None:
    super().__init__()

  def build(self, x):
    x = x.to_global(sbp=flow.sbp.broadcast())
    return x

rank = flow.env.get_rank()
t = flow.ones(4, 4) * (rank + 1)
# t = flow.ones(1) * (rank + 1)
t = t.to_global(placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.partial_sum())
print(t)
print(t.to_local())

g = GraphModel()
y = g(t)
print(y)
print(y.to_local())
