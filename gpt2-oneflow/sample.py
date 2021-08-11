
import oneflow as flow
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = flow.topk(logits, k)
    min_values = values[:, -1]
    return flow.where(logits < min_values, flow.ones_like(logits).to(logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=False):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = flow.tensor(context, device=device, dtype=flow.long).unsqueeze(0).repeat((batch_size, 1))
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = flow.zeros((batch_size, 1), device=device, dtype=flow.long)
        context.fill_(start_token)
    prev = context
    output = context
    past_key_values = None

    with flow.no_grad():
        for i in trange(length):
            if i < 10:
                print(prev)
            logits, past_key_values = model(prev, None, None, None, past_key_values, True)
            logits = logits[:, -1, :] / temperature
            # logits = top_k_logits(logits, k=top_k)

            probs = logits.softmax(dim=-1)
            prev = logits.argmax(dim=-1)
            # if sample:
            #     prev = flow.multinomial(probs, num_samples=1)   # 暂不支持采样
            # else:
            _, prev = flow.topk(probs, k=1, dim=-1)
            output = flow.cat((output, prev.to(output.dtype)), dim=1)
    return output
