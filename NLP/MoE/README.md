# The Sparsely Gated Mixture of Experts Layer for Oneflow



![source: https://techburst.io/outrageously-large-neural-network-gated-mixture-of-experts-billions-of-parameter-same-d3e901f2fe05](https://miro.medium.com/max/1000/1*AaBzgpJcySeO1UDvOQ_CnQ.png)


This repository contains the Oneflow re-implementation of the sparsely-gated MoE layer described in the paper [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) for PyTorch. 

# Usage

```python

from moe import MoE

# instantiate the MoE layer
model = MoE(input_size=1000, num_classes=20, num_experts=10,hidden_size=66, k= 4, noisy_gating=True)

# forward
y_hat, aux_loss = model(X)


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

