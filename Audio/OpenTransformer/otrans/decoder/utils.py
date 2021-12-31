import oneflow as flow

def get_transformer_decoder_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = flow.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = flow.tril(seq_mask).to(flow.int8)
    return seq_mask
