import oneflow as flow

def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False):
    world_size = 1
    init_method(weight)
    if return_master_weight:
        return weight
    return None
