
from torchvision.models.detection.image_list import ImageList
from utils.anchor_utils import AnchorGenerator
import oneflow as flow
import numpy as np
flow.enable_eager_execution()
flow.InitEagerGlobalSession()


def test_incorrect_anchors():
    incorrect_sizes = ((2, 4, 8), (32, 8), )
    incorrect_aspects = (0.5, 1.0)
    anc = AnchorGenerator(incorrect_sizes, incorrect_aspects)
    image1 = flow.Tensor(np.random.randn(3, 800, 800))
    image_list = ImageList(image1, [(800, 800)])
    feature_maps = [flow.Tensor(np.random.randn(1, 50))]


def _init_test_anchor_generator():
    anchor_sizes = ((10,),)
    aspect_ratios = ((1,),)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


def get_features(images):
    s0, s1 = images.shape[-2:]
    features = [flow.Tensor(np.random.rand(2, 8, s0 // 5, s1 // 5))]

    return features

def test_anchor_generator():
    images = flow.Tensor(np.random.randn(2, 3, 15, 15))
    features = get_features(images)
    image_shapes = [i.shape[-2:] for i in images]
    images = ImageList(images, image_shapes)

    model = _init_test_anchor_generator()
    model.eval()
    anchors = model(images, features)

    # Estimate the number of target anchors
    grid_sizes = [f.shape[-2:] for f in features]
    num_anchors_estimated = 0
    for sizes, num_anchors_per_loc in zip(grid_sizes, model.num_anchors_per_location()):
        num_anchors_estimated += sizes[0] * sizes[1] * num_anchors_per_loc

    anchors_output = flow.Tensor([[-5., -5., 5., 5.],
                                   [0., -5., 10., 5.],
                                   [5., -5., 15., 5.],
                                   [-5., 0., 5., 10.],
                                   [0., 0., 10., 10.],
                                   [5., 0., 15., 10.],
                                   [-5., 5., 5., 15.],
                                   [0., 5., 10., 15.],
                                   [5., 5., 15., 15.]])
    anchors_output = anchors_output.unsqueeze(1)
    assert num_anchors_estimated == 9
    assert len(anchors) == 2
    assert tuple(anchors[0].shape) == (9, 1, 4)
    assert tuple(anchors[1].shape) == (9, 1, 4)
    assert(np.allclose(anchors[0].numpy(), anchors_output.numpy(), 1e-5, 1e-5))
    assert (np.allclose(anchors[1].numpy(), anchors_output.numpy(), 1e-5, 1e-5))
    # assert_equal(anchors[0], anchors_output)
    # assert_equal(anchors[1], anchors_output)

if __name__ == '__main__':
    test_anchor_generator()
    # test_incorrect_anchors()