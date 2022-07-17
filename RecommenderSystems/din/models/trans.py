#保存模型代码
import os
import shutil
import tempfile
import paddle.fluid as fluid
import numpy as np
import oneflow as flow
def _load_state(path):
    """
    记载paddlepaddle的参数
    :param path:
    :return:
    """
    print(path)
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        print(dst)
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        print('else')
        state = fluid.io.load_program_state(path)
    return state

    # key_dict = {
    #     "hist_item_emb_attr.one_embedding.shadow": "hist_item_emb_attr.weight.npy",
    #     "hist_cat_emb_attr.one_embedding.shadow": "hist_cat_emb_attr.weight.npy",
    #     "target_item_seq_emb_attr.one_embedding.shadow": "target_item_emb_attr.weight.npy",
    #     "target_cat_seq_emb_attr.one_embedding.shadow": "target_cat_emb_attr.weight.npy",
    #     "target_item_emb_attr.one_embedding.shadow": "target_item_seq_emb_attr.weight.npy",
    #     "target_cat_emb_attr.one_embedding.shadow": "target_cat_seq_emb_attr.weight.npy",
    #     "item_b_attr.one_embedding.shadow": "item_b_attr.weight.npy",
    #     "attention_layer.linear_layers.0.weight": "linear_0.weight.npy",
    #     "attention_layer.linear_layers.0.bias": "linear_0.weight.npy",
    #     "attention_layer.linear_layers.2.weight": "linear_1.weight.npy",
    #     "attention_layer.linear_layers.4.weight": "linear_2.weight.npy",
    #     "first_con_layer.linear_layers.0.weight": "linearCon.weight.npy",
    #     "second_con_layer.linear_layers.0.weight": "linearCon_0.weight.npy",
    #     "second_con_layer.linear_layers.2.weight": "linearCon_1.weight.npy",
    #     "second_con_layer.linear_layers.4.weight": "linearCon_2.weight.npy",
    # }


def npy_to_state_dict(model_load_dir):
    print(f"Loading model from {model_load_dir}")
    key_dict = {
        "hist_item_emb_attr.one_embedding.shadow": "hist_item_emb_attr.weight.npy",
        "hist_cat_emb_attr.one_embedding.shadow": "hist_cat_emb_attr.weight.npy",
        "target_item_seq_emb_attr.one_embedding.shadow": "target_item_emb_attr.weight.npy",
        "target_cat_seq_emb_attr.one_embedding.shadow": "target_cat_emb_attr.weight.npy",
        "target_item_emb_attr.one_embedding.shadow": "target_item_seq_emb_attr.weight.npy",
        "target_cat_emb_attr.one_embedding.shadow": "target_cat_seq_emb_attr.weight.npy",
        "item_b_attr.one_embedding.shadow": "item_b_attr.weight.npy",
        "attention_layer.linear_layers.0.weight": "linear_0.weight.npy",
        "attention_layer.linear_layers.0.bias": "linear_0.bias.npy",
        "attention_layer.linear_layers.2.weight": "linear_1.weight.npy",
        "attention_layer.linear_layers.2.bias": "linear_1.bias.npy",
        "attention_layer.linear_layers.4.weight": "linear_2.weight.npy",
        "attention_layer.linear_layers.4.bias": "linear_2.bias.npy",
        "first_con_layer.linear_layers.0.weight": "linearCon.weight.npy",
        "first_con_layer.linear_layers.0.bias": "linearCon.bias.npy",
        "second_con_layer.linear_layers.0.weight": "linearCon_0.weight.npy",
        "second_con_layer.linear_layers.0.bias": "linearCon_0.bias.npy",
        "second_con_layer.linear_layers.2.weight": "linearCon_1.weight.npy",
        "second_con_layer.linear_layers.2.bias": "linearCon_1.bias.npy",
        "second_con_layer.linear_layers.4.weight": "linearCon_2.weight.npy",
        "second_con_layer.linear_layers.4.bias": "linearCon_2.bias.npy",
    }
    state_dict = {k: np.load(os.path.join(model_load_dir, v)) for k, v in key_dict.items()}
    init_model_path = model_load_dir + '/with_emb/'
    print(init_model_path)
    if not os.path.exists(init_model_path):
        os.mkdir(init_model_path)
    flow.save(state_dict, init_model_path)


if __name__=="__main__":
    #加载paddlepaddle的模型
    path = './0/rec'
    paddlepaddle_state = _load_state(path)
    print([k for k, v in paddlepaddle_state.items()])
    if not os.path.exists('./0/rec/'):
        os.mkdir('./0/rec/')
    for k, v in paddlepaddle_state.items():
        if "@@" not in k:
            if "emb" in k:
                name = path + '/' + k + '.npy'
                print(k)
                print(v, v.shape)
                # np.save(name, v.T)
                np.save(name, v)
            else:
                name = path + '/' + k + '.npy'
                print(k)
                print(v, v.shape)
                np.save(name, v.T)
                # np.save(name, v)
    npy_to_state_dict('./0/rec')
