import math
import logging
import oneflow as flow
import oneflow.nn as nn

logger = logging.getLogger(__name__)


class BasedAttention(nn.Module):
    def __init__(self, source_dim, output_dim, enable_output_proj=True, dropout=0.0):
        super(BasedAttention, self).__init__()

        self.enable_output_proj = enable_output_proj
        if self.enable_output_proj:
            self.output_proj = nn.Linear(source_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_context(self, values, scores, mask=None):
        """
        Args:
            values: [b, t2, v] or [b, nh, t2, v]
            scores: [b, t1, t2] or [b, nh, t1, t2]
            mask: [b, t1, t2] or [b, 1/nh, t1, t2]
        """
        assert values.dim() == scores.dim()

        if mask is not None:
            scores=flow.masked_fill(scores,mask==0, -float('inf'))
        
        weights = flow.softmax(scores, dim=-1)
        context = flow.matmul(weights, values)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        if self.enable_output_proj:
            context = self.output_proj(context)

        return self.dropout(context), weights


class MultiHeadedSelfAttention(BasedAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, share_qvk_proj=False):
        super(MultiHeadedSelfAttention, self).__init__(d_model, d_model, enable_output_proj=True, dropout=dropout_rate)

        self.d_model = d_model
        self.share_qvk_proj = share_qvk_proj
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.qvk_proj = nn.Linear(d_model, d_model if self.share_qvk_proj else d_model * 3)

    def forward(self, x, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = flow.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = flow.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, x, mask, cache=None):

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value =flow.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = flow.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights, cache


class MultiHeadedCrossAttention(BasedAttention):
    def __init__(self, n_heads, d_model, memory_dim, dropout_rate=0.0, share_vk_proj=False):
        super(MultiHeadedCrossAttention, self).__init__(d_model, d_model, enable_output_proj=True, dropout=dropout_rate)

        self.d_model = d_model
        self.share_vk_proj = share_vk_proj
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.vk_proj = nn.Linear(memory_dim, d_model if self.share_vk_proj else d_model * 2)

    def forward(self, query, memory, memory_mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = flow.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = flow.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights

    def inference(self, query, memory, memory_mask, cache=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = flow.split(memory, self.d_model, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = flow.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))
        return context, attn_weights, cache


class MultiHeadedSelfAttentionWithRelPos(BasedAttention):
    def __init__(self, n_heads, d_model, dropout_rate=0.0, skip_term_b=False, share_qvk_proj=False):
        super(MultiHeadedSelfAttentionWithRelPos, self).__init__(n_heads, d_model, dropout_rate, share_qvk_proj)

        self.d_model = d_model
        self.share_qvk_proj = share_qvk_proj
        self.skip_term_b = skip_term_b
        self.nheads = n_heads
        self.d_k = d_model // n_heads

        self.qvk_proj = nn.Linear(d_model, d_model if self.share_qvk_proj else d_model * 3)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.posu = nn.Parameter(flow.Tensor(1, 1, n_heads, self.d_k))
        self.posv = nn.Parameter(flow.Tensor(1, 1, n_heads, self.d_k))

    def _RelPosBias(self, content, abs_pos):
        """Compute relative positinal encoding.
        Args:
            content: [B, T, N, H] if not self.skip_term_b else [1, 1, N, H] oneflow.Size([16, 169, 4, 96])
            abs_pos: [B, N, S=2T-1, H] oneflow.Size([1, 4, 337, 96])
        Returns:
            torch.Tensor: Output tensor.
        """
        B, _, N, _ = content.size()
        S= abs_pos.size(2)
        T = (S + 1) // 2

        if not self.skip_term_b:
            matrix_bd = flow.matmul(content.transpose(1, 2), abs_pos.transpose(-2, -1).repeat(B,1,1,1))
        else:
            matrix_bd = flow.matmul(content.transpose(
                1, 2), abs_pos.transpose(-2, -1).repeat(B, 1, 1, 1))

        rel_pos = flow.arange(0, T, dtype=flow.long, device=matrix_bd.device)
        rel_pos = (rel_pos[None] - rel_pos[:, None]).reshape(1, 1, T, T) + (T - 1)
        return flow.gather(matrix_bd, dim=3, index=rel_pos.repeat(B, N, 1, 1))

    def forward(self, x, mask, pos):
        """
        Args:
            x: [B, T, V]
            mask: [B, 1, T]
            pos: positional embedding [B, S=2T-1, V] 
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = flow.split(x, self.d_model, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        bpos = pos.size(0)
        pos = self.pos_proj(pos).reshape(bpos, -1, self.nheads, self.d_k).transpose(1, 2)

        query_with_bias_u = query + self.posu
        query_with_bias_u = query_with_bias_u.transpose(1, 2)
        matrix_ac = flow.matmul(query_with_bias_u, key.transpose(-2, -1))

        matrix_bd = self._RelPosBias(query + self.posv if not self.skip_term_b else self.posv, pos)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, inputs, mask, pos, cache=None):
        context, attn_weights = self.forward(inputs, mask, pos)
        return context, attn_weights, cache
