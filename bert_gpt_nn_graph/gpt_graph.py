from gpt_module import * 
from gpt_optimizer import *
from gpt_data import *
# import distribute as dist_cfg
from config import get_args

class GPTTrainGraph(nn.Graph):
    def __init__(self):
        super().__init__()
        self.args = get_args()

        # dist_util = dist_cfg.get_dist_util()
        # Graph的配置
        self.train(True)
        # if self.args.fp16:
        #     self.enable_auto_mixed_precision(True) # amp
        # self.prune_parallel_cast_ops(True)
        # self.fuse_add_to_output(True)
        # self.fuse_model_update_ops(True)
        # self.fuse_cast_scale(True)
        # # turn on this flag when match ZeRO & DeepSpeed
        # self.non_distributed_optimizer(False)
        # if self.args.num_accumulation_steps > 1:
        #     self.num_gradient_accumulation_steps(self.args.num_accumulation_steps)

        # ConsistentModule的注册
        # TODO():改为ConsistentModule的写法
        self.data_loader = GPTDataLoader()
        self.gpt_model = GPTModel()
        self.loss = SparseSoftmaxCrossEntropyLoss()

        # 配置ConsistentModule的分布式信息
        # self.data_loader.src.to(flow.placement("cpu"))
        # self.data_loader.src.stage(dist_util.get_layer_stage(0))
        # self.data_loader.data.to(flow.placement(dist_util.get_layer_placement(0)))
        # self.data_loader.data.stage(dist_util.get_layer_stage(0))
        # self.data_loader.label.to(flow.placement(dist_util.get_layer_placement(-1)))
        # self.data_loader.label.stage(dist_util.get_layer_stage(-1))

        # emb_node = self.gpt_model.embedding
        # emb_node.to(flow.placement(dist_util.get_layer_placement(0)))
        # # emb_node.amp_white_identity() # ?
        # emb_node.stage(dist_util.get_layer_stage(0))
        # emb_node.wpe.parallel_distribution(dist_cfg.get_wpe_parallel_dist()) # configure parameter
        # emb_node.wpe.amp_white_identity() # ?
        # emb_node.wte.parallel_distribution(dist_cfg.get_wte_parallel_dist()) # configure parameter
        # emb_node.wte.amp_white_identity() # ?

        # transformer_node = self.gpt_model.transformer
        # if transformer_node._origin.multihead_attention_fusion:
        #     trans_permute_node = transformer_node.permute
        #     trans_permute_node.to(flow.placement(dist_util.get_layer_placement(0))
        #     trans_permute_node.stage(dist_util.get_layer_stage(0))
        
        # for i in range(transformer_node._origin.num_layers):
        #     layer_node = transformer_node.layers[i]

        #     layer_node.to(flow.placement(dist_util.get_layer_placement(i))
        #     layer_node.stage(dist_util.get_layer_stage(i))

        #     if args.checkpoint_activations:
        #         layer_node.checkpoint_activations(True)

        #     norm1_node = layer_node.norm1
        #     if norm1_node._origin.elementwise_affine:
        #         norm1_node.weight.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())
        #         norm1_node.bias.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())

        #     norm2_node = layer_node.norm2
        #     if norm2_node._origin.elementwise_affine:
        #         norm2_node.weight.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())
        #         norm2_node.bias.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())

        #     attn_node = layer_node.attn
        #     attn_node.col_linear.w.parallel_distribution(dist_util.get_col_linear_weight_parallel_dist())
        #     attn_node.col_linear.b.parallel_distribution(dist_util.get_col_linear_bias_parallel_dist())
        #     # grad_sbp 写法1
        #     attn_node.col_linear.inputs[0].grad_sbp = ["S(0)", "B"]
        #     attn_node.row_linear.w.parallel_distribution(dist_util.get_row_linear_weight_parallel_dist())
        #     attn_node.row_linear.b.parallel_distribution(dist_util.get_row_linear_bias_parallel_dist())

        #     mlp_node = layer_node.mlp
        #     mlp_node.col_linear.w.parallel_distribution(dist_util.get_col_linear_weight_parallel_dist())
        #     mlp_node.col_linear.b.parallel_distribution(dist_util.get_col_linear_bias_parallel_dist())
        #     mlp_node.row_linear.w.parallel_distribution(dist_util.get_row_linear_weight_parallel_dist())
        #     mlp_node.row_linear.b.parallel_distribution(dist_util.get_row_linear_bias_parallel_dist())

        # trans_layer_norm_node = transformer_node.layer_norm
        # trans_layer_norm_node.to(flow.placement(dist_util.get_layer_placement(-1)))
        # trans_layer_norm_node.stage(dist_util.get_layer_stage(-1))
        # if trans_layer_norm_node._origin.elementwise_affine:
        #     trans_layer_norm_node.weight.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())
        #     trans_layer_norm_node.bias.parallel_distribution(dist_util.get_layernorm_params_parallel_dist())


        # logits_node = self.gpt_model.logits
        # logits_node.to(flow.placement(dist_util.get_layer_placement(-1)))
        # logits_node.stage(dist_util.get_layer_stage(-1))

        # loss_node = self.loss
        # loss_node.to(flow.placement(dist_util.get_layer_placement(-1)))
        # loss_node.stage(dist_util.get_layer_stage(-1))
        # if dist_util.tensor_model_parallel_size <= 1:
        #     loss_node.loss_module.outputs[0].amp_white_identity() # ?

        # 添加optimizer
        # opt = make_optimizer(self.args, self.gpt_model.parameters())
        # self.add_optimizer(optimizer=opt)

   
    # 使用ConsistentModule构建Graph
    def build(self) :
        data, labels = self.data_loader()
        logits = self.gpt_model(data)
        loss = self.loss(logits, labels)
        loss.backward()
        # 总是返回lazy tensor，用户自己决定的对lazy tensor的值的获取
        # NOTE(): 聚合loss结果，用于在signle-client打印和校验
        # losses = distribute.output_parallel_cast(losses)
        return loss # oneflow.LazyTensor
    
    
def main():
    gpt_train_graph = GPTTrainGraph()

    loss = gpt_train_graph() # 被call后才会真正执行construct，这时才使用consistant_cast的配置

if __name__ == "__main__":
    main()


    