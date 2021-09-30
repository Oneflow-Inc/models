
import copy
import logging
import oneflow as flow
# logger = logging.getLogger(__name__)

def convert_pt_checkpoint_to_of(model, pt_checkpoint_path="gpt2-pytorch_model.bin", of_checkpoint_path="gpt2_oneflow_model"):
    import torch
    parameters = torch.load(pt_checkpoint_path)
    new_parameters = {}
    keys_to_ignore = [
        "transformer.h.0.attn.bias", "transformer.h.0.attn.masked_bias",
        "transformer.h.1.attn.bias", "transformer.h.1.attn.masked_bias",
        "transformer.h.2.attn.bias", "transformer.h.2.attn.masked_bias",
        "transformer.h.3.attn.bias", "transformer.h.3.attn.masked_bias",
        "transformer.h.4.attn.bias", "transformer.h.4.attn.masked_bias",
        "transformer.h.5.attn.bias", "transformer.h.5.attn.masked_bias",
        "transformer.h.6.attn.bias", "transformer.h.6.attn.masked_bias",
        "transformer.h.7.attn.bias", "transformer.h.7.attn.masked_bias",
        "transformer.h.8.attn.bias", "transformer.h.8.attn.masked_bias",
        "transformer.h.9.attn.bias", "transformer.h.9.attn.masked_bias",
        "transformer.h.10.attn.bias", "transformer.h.10.attn.masked_bias",
        "transformer.h.11.attn.bias", "transformer.h.11.attn.masked_bias",
    ]
    for key, value in parameters.items():
        if key in keys_to_ignore:
                continue
        if "num_batches_tracked" not in key:
            val = value.detach().cpu().numpy()
            new_parameters[key] = val
    model.load_state_dict(new_parameters, strict=False)
    # model.tie_embeddings()
    flow.save(model.state_dict(), of_checkpoint_path)
