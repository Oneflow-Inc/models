# Res-EnDe-LSTM Algorithm

# Requirements

```
Python 3.6+
Numpy
OneFlow
```
# 1.Score 
Score:  48.86911847

# 2、目录结构
```
|-- figure # 模型训练过程指标下降情况图示

|-- model_para # 模型参数

|-- log # 用OneFlow的在线开发好像就会有这个目录（可忽略）

|-- README.md # 项目说明文件

|-- data # 数据集文件夹，包括infer的结果文件，用作参考的结果文件，和记录训练过程的数据流文件（一个txt）

|-- dataset.npz # 数据集

|-- run.sh # 模型运行和推理的脚本，可直接生成测试结果npz的文件(里面就一句话：python train.py)

|-- infer.py # 推理文件

|-- train.py # 模型训练文件，训练完成后会自动推理出结果
```
github这里只是代码，没有数据文件，因为有一些文件夹因为文件过大的原因不能传上去，所以在此给一个完整版的下载地址：

“链接：https://pan.baidu.com/s/1KJU-ndq0E_vfEj-fmw7I-A 
提取码：1eco”

# 3、代码启动脚本或启动方式

## 3.1 Train+Test：  python train.py

这个train文件会先加载为预先训练好的模型参数，然后训练160次整个训练集，之后另外保存模型参数，然后输出模型结果。这个是训练和infer合在一起的。

考虑到训练时间过长，这里提供直接输出结果的方法

## 3.2 单独Test python test.py
这里预先训练好的模型参数也是用这个train.py文件训出来的，所以在test.py文件里预定义好了模型的构成，然后程序会自动加载我预先存好的模型参数，然后直接infer一个test_y.npz文件出来。

之后会有一行测试输出，这个输出是我用得分比较高的npz结果文件来和程序infer出来的结果做一个对比：相关参数应该是：

test: test_rmse_loss 14.863170,test_mae_loss 9.348677,test_mape_loss 12.778813
## 3.3 包括训练和测试一键生成npz文件的脚本运行： sh run.sh（可选，最好有）

我这个run.sh不一定对，其实就是python train.py这一句话。

