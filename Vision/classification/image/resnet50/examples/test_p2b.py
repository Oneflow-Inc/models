import oneflow as flow

class P2BModule(flow.nn.Module):
  def __init__(self) -> None:
      super().__init__()

  def forward(self, t):
    t = t.to_global(placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.partial_sum())
    # p2b
    t = t.to_global(sbp=flow.sbp.broadcast())
    return t

class GraphModel(flow.nn.Graph):
  def __init__(self) -> None:
    super().__init__()
    self.p2b = P2BModule()

  def build(self, x):
    x = self.p2b(x)
    return x

rank = flow.env.get_rank()
t = flow.ones(4, 4) * (rank + 1)
print(t)

g = GraphModel()
y = g(t)
print(rank, y)



# x_split = [
#   flow.tensor([
#     [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#     [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
#   ]),
#   flow.tensor([
#     [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
#     [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
#   ]),
#   flow.tensor([
#     [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#     [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
#   ]),
#   flow.tensor([
#     [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
#     [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
#   ]),
# ]

# g = GraphModel()
# rank = flow.env.get_rank()

# x = x_split[rank]
# print(x)

# y = g(x)
# print(rank, y)













# class Model(flow.nn.Module):
#   def __init__(self) -> None:
#     super().__init__()
#     self.placement = flow.placement("cuda", [0, 1, 2, 3])
#     self.s0 = flow.sbp.split(0)
#     self.b = flow.sbp.broadcast
#     self.identity = flow.nn.Identity()

#   def forward(self, x):
#     x = self.identity(x.to_global(placement=self.placement, sbp=self.s0))
#     print(f"x {x}, \n {x.shape}\n")
    
#     x = flow.sum(x, dim=0)
#     print(f"x {x}, \n {x.shape}\n")
    
#     x = self.identity(x.to_global(placement=self.placement, sbp=self.b))
#     print(f"x {x}, \n {x.shape}\n")
#     return x

    
#     # y = flow.randn(4, 3, placement=self.placement, sbp=self.b)
#     # y = self.identity(x)
#     # print(f"y {y}, \n {y.shape}\n")
#     # return y

# m = Model().to("cuda")
# out = m(x.to("cuda"))
# print(out)
