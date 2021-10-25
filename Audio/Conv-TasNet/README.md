# Conv-TasNet

An Oneflow implementation of Conv-TasNet described in [TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454)


## Requirements

```bash
pip install -r requirements.txt
```

## Data
We train and evaluate our models on the partial data from WSJ0-2mix, which is not an open-source dataset and can be found [here](https://catalog.ldc.upenn.edu/LDC93S6A).


## Usage
```bash
bash run.sh
```
That's all!

## Visualization of loss

![loss](conv_tasnet_loss.png) 

## Result

  |         | Loss |SI-SDR(dB)|Pretrained Model |
  | :---: | :---: |  :---: | :---: |
  |   Oneflow   | -25.887|25.227|[ConvTasnet_Model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/final.pth.tar.zip)|
  |   Pytorch   | -25.727|26.127 |--- |

 
## Separation demo
You can find one separation demo from [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/Conv-Tasnet_demo), which includes an two-speaker original mixed voice and its corresponding seperating results.



## Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.
