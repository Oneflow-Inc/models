# The Sparsely Gated Mixture of Experts Layer for Oneflow



![source: https://techburst.io/outrageously-large-neural-network-gated-mixture-of-experts-billions-of-parameter-same-d3e901f2fe05](https://miro.medium.com/max/1000/1*AaBzgpJcySeO1UDvOQ_CnQ.png)


This repository contains the Oneflow re-implementation of the sparsely-gated MoE layer described in the paper [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) for PyTorch. 

# Usage



```python

# k indicates the top-k switching
model = MoE(expert_network, input_size, output_size, num_experts, noisy_gating, k)


```




# Citing
```
@misc{rau2019moe,
    title={Sparsely-gated Mixture-of-Experts PyTorch implementation},
    author={Rau, David},
    journal={https://github.com/davidmrau/mixture-of-experts},
    year={2019}
}
```

