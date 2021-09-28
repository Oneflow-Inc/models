# CCF BDCI BERT系统调优赛题baseline

此baseline基于Oneflow的静态图模块Graph进行实现，在利用eager动态图方便调试的优势将模型debug完成后，通过自定义 Graph 类将其转成静态图，具体教程可以参考[官方上手文档](https://docs.oneflow.org/v0.5.0/basics/08_nn_graph.html)。

由于此题涉及到单机2卡的系统调优，因此可能会需要使用到ddp，因此也提供了ddp的baseline例子。有关ddp的其它介绍，可以参照[文档](https://docs.oneflow.org/v0.5.0/parallelism/05_ddp.html)。

## 使用说明

1. 准备好工作环境（前置要求：已在环境中安装好[Oneflow](https://github.com/Oneflow-Inc/oneflow)）
    
    ```bash
    git clone -b bert_competition_baseline_graph https://github.com/Oneflow-Inc/models
    cd models/bert-oneflow/
    ```
    
2. 运行train_BERT_base.sh和train_BERT_large.sh 单机单卡的baseline。保持其它参数不变，通过调节shell文件里的hidden_size参数，即可观察不同hidden_size所占显存的变化（可通过`watch -n 0.1 nvidia-smi`直观观察）
    
    ```bash
    vim train_BERT_large.sh
    #...
    
    python3 run_pretraining.py \
      --ofrecord_path $OFRECORD_PATH \
      --checkpoint_path $CHECKPOINT_PATH \
      --lr $LEARNING_RATE \
      --epochs $EPOCH \
      --train-batch-size $TRAIN_BATCH_SIZE \
      --val-batch-size $VAL_BATCH_SIZE \
      --seq_length=512 \
      --max_predictions_per_seq=80 \
      --num_hidden_layers=24 \
      --num_attention_heads=16 \
      --hidden_size=1024 \ #要调节的参数
      --max_position_embeddings=512 \
      --type_vocab_size=2 \
      --vocab_size=30522 \
      --attention_probs_dropout_prob=0.1 \
      --hidden_dropout_prob=0.1
    
    sh train_BERT_large.sh
    ```
    
3. 目前也已经提供了ddp运行脚本，默认的配置为1机2卡，通过train_BERT_large_ddp.sh运行ddp的例子：

```bash
vim train_BERT_large_ddp.sh
#...

python3 -m oneflow.distributed.launch \
  --nproc_per_node 2 \ #单个节点的进程数
  --nnodes 1 \ #节点个数
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 17789 \
#...

sh train_BERT_large.sh
```
