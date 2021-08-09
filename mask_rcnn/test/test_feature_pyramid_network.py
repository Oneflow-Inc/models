from collections import OrderedDict
import oneflow as flow
from ops.feature_pyramid_network import FeaturePyramidNetwork
import numpy as np
flow.enable_eager_execution()
flow.InitEagerGlobalSession()



def test_fpn():
    m = FeaturePyramidNetwork([10, 20, 30], 5)
    x = OrderedDict()
    x['feat0'] = flow.Tensor(np.random.rand(1, 10, 64, 64))
    x['feat2'] = flow.Tensor(np.random.rand(1, 20, 16, 16))
    x['feat3'] = flow.Tensor(np.random.rand(1, 30, 8, 8))
    output = m(x)
    print([(k, v.shape) for k, v in output.items()])
    assert(output['feat0'].shape == flow.Size([1, 5, 64, 64]))
    assert(output['feat2'].shape == flow.Size([1, 5, 16, 16]))
    assert(output['feat3'].shape == flow.Size([1, 5, 8, 8]))



if __name__ == '__main__':
    test_fpn()