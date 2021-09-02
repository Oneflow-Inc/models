# 问题说明

使用 Transformer 模型在 GPU 上训练针对 seq2seq 任务的模型时无法收敛。在 CPU 上运行时正常，使用手动实现的 ``LayerNorm`` 时正常。具体原因未知。 Transformer 模型的实现在`transformer` 文件夹中，任务代码在 `odd_numbers` 文件夹中，通过 `train.sh` 进行训练。 在 `transformer\transformer.py` 中切换 ``LayerNorm`` 的版本。

### GPU 运行结果 

```bash
epoch: 1 train loss: 9.259610312325615
epoch: 1 val loss: 8.969719568888346
epoch: 2 train loss: 7.734267221178327
epoch: 2 val loss: 8.523958206176758
0.0
epoch: 3 train loss: 7.009487778799874
epoch: 3 val loss: 8.19363816579183
epoch: 4 train loss: 6.496719060625349
epoch: 4 val loss: 8.011690775553385
epoch: 5 train loss: 6.669434193202427
epoch: 5 val loss: 7.976550579071045
0.1
epoch: 6 train loss: 6.4799286706107
epoch: 6 val loss: 7.902670224507649
epoch: 7 train loss: 6.299545274462019
epoch: 7 val loss: 7.782184759775798
epoch: 8 train loss: 6.377702140808106
epoch: 8 val loss: 7.797385851542155
0.0
epoch: 9 train loss: 6.4196480342320035
epoch: 9 val loss: 7.7692036628723145
epoch: 10 train loss: 6.200791154588972
epoch: 10 val loss: 7.700966517130534
epoch: 11 train loss: 6.0659737859453475
epoch: 11 val loss: 7.612149556477864
0.1
```

### CPU 运行结果

```bash
epoch: 1 train loss: 9.036801746913365
epoch: 1 val loss: 8.52875550587972
epoch: 2 train loss: 6.852300943647112
epoch: 2 val loss: 7.647552331288655
0.2
epoch: 3 train loss: 5.157164055960519
epoch: 3 val loss: 6.938950538635254
epoch: 4 train loss: 3.780674457550049
epoch: 4 val loss: 6.247670968373616
epoch: 5 train loss: 2.6901396342686246
epoch: 5 val loss: 5.785963217417399
0.4
epoch: 6 train loss: 1.8703475849969047
epoch: 6 val loss: 5.266840934753418
epoch: 7 train loss: 1.307879672731672
epoch: 7 val loss: 4.861152807871501
epoch: 8 train loss: 0.9393650531768799
epoch: 8 val loss: 4.51524289449056
0.4
epoch: 9 train loss: 0.6906436783926827
epoch: 9 val loss: 4.257654984792073
epoch: 10 train loss: 0.5165321384157453
epoch: 10 val loss: 4.160113016764323
epoch: 11 train loss: 0.3933414212294987
epoch: 11 val loss: 3.99604860941569
0.8
```

### 使用手动实现的 LayerNorm 在 GPU 上的运行结果

```bash
epoch: 1 train loss: 9.016217286246164
epoch: 1 val loss: 8.534285545349121
epoch: 2 train loss: 6.865990938459124
epoch: 2 val loss: 7.689682960510254
0.1
epoch: 3 train loss: 5.258445726122175
epoch: 3 val loss: 6.969688097635905
epoch: 4 train loss: 4.002006353650774
epoch: 4 val loss: 6.448536078135173
epoch: 5 train loss: 3.00015549659729
epoch: 5 val loss: 5.947791576385498
0.7
epoch: 6 train loss: 2.2113828250340055
epoch: 6 val loss: 5.467899322509766
epoch: 7 train loss: 1.6332548345838274
epoch: 7 val loss: 5.116685549418132
epoch: 8 train loss: 1.2277311699731008
epoch: 8 val loss: 4.814037322998047
0.4
epoch: 9 train loss: 0.9367656741823469
epoch: 9 val loss: 4.570450146993001
epoch: 10 train loss: 0.7232056191989353
epoch: 10 val loss: 4.38416846593221
epoch: 11 train loss: 0.5687279718262809
epoch: 11 val loss: 4.171596050262451
0.8
```
