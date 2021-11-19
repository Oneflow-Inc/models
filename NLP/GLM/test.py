import oneflow
import oneflow as flow
from modeling import GLMModel

if __name__ == "__main__":
    model = GLMModel(num_layers=12,
                     vocab_size=123123,
                     hidden_size=512,
                     num_attention_heads=8,
                     embedding_dropout_prob=0.1,
                     attention_dropout_prob=0.1,
                     output_dropout_prob=0.1,
                     max_sequence_length=256,
                     max_memory_length=512,
                     checkpoint_activations=False,
                     checkpoint_num_layers=12,
                     parallel_output=True,
                     relative_encoding=False,
                     block_position_encoding=True,
                     output_predict=True,
                     spell_length=123,
                     spell_func=None,
                     attention_scale=1.0)

    model.cuda()

    tokens = flow.zeros(4, 332).to("cuda")
    labels = flow.zeros(4, 332).to(device="cuda", dtype=oneflow.int64)
    loss_mask = flow.zeros(4, 332).to("cuda")
    attention_mask = flow.zeros(4).to(device="cuda", dtype=oneflow.int64)
    position_ids = flow.zeros(4, 2, 332).to(device="cuda", dtype=oneflow.int64)

    logits, *mems = model(tokens, position_ids, attention_mask)
    print(logits)
