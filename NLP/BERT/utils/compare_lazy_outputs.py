#!/usr/bin/python3
import oneflow.nn as nn
import oneflow as flow
from modeling import BertForPreTraining
import pickle
import numpy as np


def get_masked_lm_loss(
    logit_blob,
    masked_lm_positions,
    masked_lm_labels,
    label_weights,
    max_prediction_per_seq=20,
):
    # gather valid position indices
    logit_blob = flow.gather(
        logit_blob, index=masked_lm_positions.unsqueeze(2).repeat(1, 1, 30522), dim=1,
    )
    logit_blob = flow.reshape(logit_blob, [-1, 30522])
    label_id_blob = flow.reshape(masked_lm_labels, [-1])

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    pre_example_loss = nn.CrossEntropyLoss(reduction="none")(logit_blob, label_id_blob)
    pre_example_loss = flow.reshape(pre_example_loss, [-1, max_prediction_per_seq])
    sum_label_weight = flow.sum(label_weights, dim=-1)
    sum_label_weight = sum_label_weight / label_weights.shape[0]
    numerator = flow.sum(pre_example_loss * label_weights)
    denominator = flow.sum(label_weights) + 1e-5
    loss = numerator / denominator
    return logit_blob, loss


def change_name_from_lazy_to_eager(lazy_name: str):
    eager_name = lazy_name.replace("-", ".").replace("layer_", "layer.")
    # In lazy model, params in layernorm are named as `gamma` and `beta`,
    # but in eager model, these params are named as `weight` and `bias`
    if "LayerNorm" in eager_name:
        eager_name = eager_name.replace("gamma", "weight").replace("beta", "bias")
    return eager_name


def load_params_from_lazy(eager_state_dict, lazy_model_path):
    print(f"Restroing model from {lazy_model_path}")
    lazy_state_dict = flow.load(lazy_model_path)
    all_eager_names_list = set(eager_state_dict.keys())

    # load regular weights
    for lazy_name, lazy_weight in lazy_state_dict.items():
        # skip momentum and momentum^2 for optimizer
        if lazy_name.endswith("-v") or lazy_name.endswith("-m"):
            continue
        eager_name = change_name_from_lazy_to_eager(lazy_name)
        if eager_name not in all_eager_names_list:
            print(f"{eager_name} is not matched")
            continue
        else:
            all_eager_names_list.remove(eager_name)
            if (
                ("dense.weight" in eager_name)
                or ("query.weight" in eager_name)
                or ("value.weight" in eager_name)
                or ("key.weight" in eager_name)
            ):
                lazy_weight = flow.tensor(lazy_weight.numpy().transpose())
            eager_state_dict[eager_name].data.copy_(lazy_weight)

    # load embedding
    eager_state_dict["bert.embeddings.word_embeddings.weight"].data.copy_(
        lazy_state_dict["bert-embeddings-word_embeddings"]
    )
    eager_state_dict["bert.embeddings.token_type_embeddings.weight"].data.copy_(
        lazy_state_dict["bert-embeddings-token_type_embeddings"]
    )
    eager_state_dict["bert.embeddings.position_embeddings.weight"].data.copy_(
        flow.tensor(
            lazy_state_dict["bert-embeddings-position_embeddings"].numpy().squeeze(0)
        )
    )


if __name__ == "__main__":
    lazy_model_path = "./of_bert_1000000_model_log/snapshot_snapshot_1000000"

    bert_module = BertForPreTraining(
        30522, 128, 768, 12, 12, 3072, nn.GELU(), 0.0, 0.0, 512, 2,
    )

    load_params_from_lazy(bert_module.state_dict(), lazy_model_path)

    assert id(bert_module.cls.predictions.decoder.weight) == id(
        bert_module.bert.embeddings.word_embeddings.weight
    )

    with open(
        "../../OneFlow-Benchmark/LanguageModeling/BERT/lazy_input_output_1.pickle", "rb"
    ) as handle:
        lazy_info = pickle.load(handle)

    total_loss = lazy_info["total_loss"]
    mlm_loss = lazy_info["mlm_loss"]
    nsp_loss = lazy_info["nsp_loss"]
    mlm_logit_prob = lazy_info["mlm_logit_prob"]
    ns_logit_prob = lazy_info["ns_logit_prob"]
    input_ids = lazy_info["input_ids"]
    next_sentence_labels = lazy_info["next_sentence_labels"]
    input_mask = lazy_info["input_mask"]
    segment_ids = lazy_info["segment_ids"]
    masked_lm_ids = lazy_info["masked_lm_ids"]
    masked_lm_positions = lazy_info["masked_lm_positions"]
    masked_lm_weights = lazy_info["masked_lm_weights"]

    bert_module.to("cuda")

    prediction_scores, seq_relationship_scores = bert_module(
        flow.tensor(input_ids).to("cuda"),
        flow.tensor(segment_ids).to("cuda"),
        flow.tensor(input_mask).to("cuda"),
    )

    next_sentence_loss = nn.CrossEntropyLoss()(
        seq_relationship_scores.view(-1, 2),
        flow.tensor(next_sentence_labels).view(-1).to("cuda"),
    )

    logit_prob, masked_lm_loss = get_masked_lm_loss(
        prediction_scores,
        flow.tensor(masked_lm_positions).to("cuda"),
        flow.tensor(masked_lm_ids).to("cuda"),
        flow.tensor(masked_lm_weights).to("cuda"),
    )

    eager_total_loss = next_sentence_loss + masked_lm_loss

    # Loss equal
    assert np.allclose(total_loss, eager_total_loss.numpy()), "total loss is not equal!"
    assert np.allclose(
        mlm_loss, masked_lm_loss.numpy()
    ), "masked language model loss is not equal!"
    assert np.allclose(
        nsp_loss, next_sentence_loss.numpy()
    ), "next sentence loss is not equal!"

    # Output equal
    assert np.allclose(
        mlm_logit_prob, logit_prob.numpy(), rtol=1e-4, atol=1e-4
    ), "masked head output loss is not equal!"
    assert np.allclose(
        ns_logit_prob, seq_relationship_scores.view(-1, 2).numpy()
    ), "next sentence output head is not equal!"

    print("All values are matched!")
