import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser("flags for draw image")
    parser.add_argument(
        "--txt_root",
        type=str,
        default="./results/check_info/",
        help="your txt root dir",
    )
    parser.add_argument(
        "--save_root", type=str, default="./loss_txt/", help="your draw image save dir",
    )
    return parser.parse_args()


# helpers
def load_data(file_path):
    def load_txt(file_path):
        data = []
        with open(file_path, "r") as f:
            for _line in f.readlines():
                data.append(float(_line.strip()))
        return data

    if type(file_path) == type([]):
        total = 0
        for fp in file_path:
            total += np.array(load_txt(fp))
        total = total / len(file_path)
        return total.tolist()
    else:
        return load_txt(file_path)


def draw_and_save(info_dic):
    # info_dic: {
    #   "title": "compare_loss"
    #   "save_path": "your_save_path"
    #   "txts": [a.txt, b.txt],
    #   "names": [a_name, b_name],
    #   "xlabel": "epochs",
    #   "ylabel": "acc",
    #   "xlim": [0, 1],                  # Optional
    #   "ylim": [0, 1],                  # Optional
    #   "do_abs_minus": False            # Optional
    # }
    title, save_path = info_dic["title"], info_dic["save_path"]
    txts, labels = info_dic["txts"], info_dic["names"]
    xlabel, ylabel = info_dic["xlabel"], info_dic["ylabel"]
    xlim, ylim = info_dic.get("xlim", 0), info_dic.get("ylim", 0)
    do_abs_minus = info_dic.get("do_abs_minus", False)
    assert len(txts) == len(labels)

    # setup
    plt.rcParams["figure.dpi"] = 100
    plt.clf()
    plt.xlabel(xlabel, fontproperties="Times New Roman")
    plt.ylabel(ylabel, fontproperties="Times New Roman")
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if do_abs_minus:
        assert len(txts) == 2
        data1, data2 = load_data(txts[0]), load_data(txts[1])
        assert len(data1) == len(data2)
        idxs = [i for i in range(len(data1))]
        abs_data = [abs(data1[i] - data2[i]) for i in idxs]
        plt.plot(idxs, abs_data)
    else:
        for txt, label in zip(txts, labels):
            data = load_data(txt)[:300]
            idxs = [i for i in range(len(data))]
            plt.plot(idxs, data, label=label)
    plt.title(title)
    plt.legend(loc="upper right", frameon=True, fontsize=8)
    plt.savefig(save_path)


def add_pth(a, b):
    return os.path.join(a, b)


if __name__ == "__main__":
    args = _parse_args()
    # txt_root = args.txt_root
    save_root = args.save_root
    # assert os.path.exists(txt_root), 'you should run "check/check.sh" before drawing graphs'

    # draw and save
    os.makedirs(save_root, exist_ok=True)
    draw_and_save(
        {
            "title": "eager_graph_ddp_vs_consistent_loss_compare",
            "save_path": add_pth(save_root, "eager_graph_4gpu_ddp_vs_consistent.png"),
            "txts": [
                # "loss_txt/bert_graph_sgd_amp_consistent_ddp_1gpu_loss.txt",
                # "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_txt/loss_info_sgd_amp_ddp_4gpu_diffpart_zwx.txt",
                # "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_txt/loss_info_sgd_amp_ddp_4gpu_shuffle_zwx.txt",
                # "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_txt/loss_info_sgd_amp_ddp_4gpu_shuffle.txt",
                # "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_info_sgd_amp_ddp_1gpu.txt",
                "loss_txt/bert_graph_sgd_amp_consistent_4gpu_4partdiff_fp32_loss.txt",
                [
                    "loss_txt/bert_4gpu_eager_consistent_diff_loss0.txt",
                    "loss_txt/bert_4gpu_eager_consistent_diff_loss1.txt",
                    "loss_txt/bert_4gpu_eager_consistent_diff_loss2.txt",
                    "loss_txt/bert_4gpu_eager_consistent_diff_loss3.txt",
                ],
            ],
            "names": [
                "graph_consistent_fp32_loss",
                "eager_ddp_loss",
            ],  # "lazy_reapeat4part_loss"
            "xlabel": "iter",
            "ylabel": "loss",
        }
    )
    # draw_and_save(
    #     {
    #         "title": "lazy_graph_adamw_lr1e-3_loss_compare",
    #         "save_path": add_pth(save_root, "lazy_graph_adam.png"),
    #         "txts": [
    #             "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_info_adamw_exclude.txt",
    #             "loss_txt/bert_graph_adamw_exclude_loss.txt",
    #         ],
    #         "names": ["lazy_loss", "graph_loss"],
    #         "xlabel": "iter",
    #         "ylabel": "loss",
    #     }
    # )

    # draw_and_save({ "title": "compare_abs_loss",
    #                 "save_path": add_pth(save_root, "compare_abs_loss.png"),
    #                 "txts": [add_pth(txt_root, "eager_losses.txt"),
    #                         add_pth(txt_root, "graph_losses.txt")],
    #                 "names": ["eager_loss", "graph_loss"],
    #                 "xlabel": "iters",
    #                 "ylabel": "abs_loss",
    #                 "do_abs_minus": True
    #                 })
