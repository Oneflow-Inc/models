import oneflow as flow
from .utils import divide, _initialize_affine_weight
import oneflow.nn.init as init
import oneflow.nn.functional as F
from oneflow.nn.parameter import Parameter

class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)

class PositionalEmbedding(flow.nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (flow._C.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = flow.ger(pos_seq, self.inv_freq)
        pos_emb = flow.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]

class VocabParallelEmbedding(flow.nn.Module):
       
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, 0,1)
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        self.weight = Parameter(flow.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        # self.weight.model_parallel = True
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        
        masked_input = (input_ - self.vocab_start_index).to(flow.int)

        if input_.is_consistent:
            weight_fp16 = flow._C.amp_white_identity(self.weight) 
        else:
            weight_fp16 = self.weight

        output_parallel = F.embedding(masked_input, 
                                      weight_fp16,
                                      self.padding_idx, 
                                      self.max_norm,
                                      None,               #self.norm_type, 暂时变为none
                                      self.scale_grad_by_freq,
                                      self.sparse)
                                      
        output = output_parallel
        
        return output