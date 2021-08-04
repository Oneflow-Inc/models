# models
Models and examples implement with OneFlow(version >= 0.4.0).

## Install Oneflow

https://github.com/Oneflow-Inc/oneflow#install-with-pip-package

## Build custom ops from source
In the root directory, run:
```bash
mkdir build
cd build
cmake ..
make -j$(nrpoc)
```
Example of using ops:
```bash
from ops import RoIAlign
pooler = RoIAlign(output_size=(14, 14), spatial_scale=2.0, sampling_ratio=2)
```
