# p2b.py
import oneflow as flow

# prepare data on each device
rank = flow.env.get_rank()
t = flow.ones(4, 4) * (rank + 1)

# set sbp partial_sum
t = t.to_global(placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.partial_sum())
print(t.to_local())

# p2b
t = t.to_global(sbp=flow.sbp.broadcast())

# show result
print(t.to_local())