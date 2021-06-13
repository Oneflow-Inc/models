## Pretrained Model Converted From PyTorch

```shell
$wget https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
```

```python
    import numpy as np
    import torch

    flow.enable_eager_execution()

    pretrained_config = {
        "shufflenetv2_x0.5": {
            "url:": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            "shape": [[4, 8, 4], [24, 48, 96, 192, 1024]],
            "file": "shufflenetv2_x0.5-f707e7126e.pth",
        },
        "shufflenetv2_x1.0": {
            "url:": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "shape": [[4, 8, 4], [24, 116, 232, 464, 1024]],
            "file": "shufflenetv2_x1-5666bf0f80.pth",
        },
    }

    for k, v in pretrained_config.items():
        m = _shufflenetv2(k, *v["shape"])

        parameters = torch.load(v["file"])
        for key, value in parameters.items():
            val = value.detach().cpu().numpy()
            parameters[key] = val
            print("key:", key, "value.shape", val.shape)

        m.load_state_dict(parameters)
        flow.save(m.state_dict(), "oneflow_" + k)
```
