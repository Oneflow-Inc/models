import os
import argparse
import matplotlib.pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser("flags for draw image")
    parser.add_argument(
        "--txt_root",
        type=str,
        default="./results/check_info/",
        help="your txt root dir",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./results/picture/",
        help="your draw image save dir",
    )
    return parser.parse_args()


# helpers
def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for _line in f.readlines():
            data.append(float(_line.strip()))
    return data


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
            data = load_data(txt)
            idxs = [i for i in range(len(data))]
            plt.plot(idxs, data, label=label)
    plt.title(title)
    plt.legend(loc="upper right", frameon=True, fontsize=8)
    plt.savefig(save_path)


def add_pth(a, b):
    return os.path.join(a, b)


if __name__ == "__main__":
    args = _parse_args()
    txt_root = args.txt_root
    save_root = args.save_root
    assert os.path.exists(
        txt_root
    ), 'you should run "check/check.sh" before drawing graphs'

    # draw and save
    os.makedirs(save_root, exist_ok=True)
    draw_and_save(
        {
            "title": "compare_acc",
            "save_path": add_pth(save_root, "acc.png"),
            "txts": [
                add_pth(txt_root, "eager_acc.txt"),
                add_pth(txt_root, "graph_acc.txt"),
            ],
            "names": ["eager_acc", "graph_acc"],
            "xlabel": "epochs",
            "ylabel": "acc",
        }
    )
    draw_and_save(
        {
            "title": "compare_loss",
            "save_path": add_pth(save_root, "compare_loss.png"),
            "txts": [
                add_pth(txt_root, "eager_losses.txt"),
                add_pth(txt_root, "graph_losses.txt"),
            ],
            "names": ["eager_loss", "graph_loss"],
            "xlabel": "epochs",
            "ylabel": "loss",
        }
    )
    draw_and_save(
        {
            "title": "compare_train_step_time",
            "save_path": add_pth(save_root, "compare_train_step_time.png"),
            "txts": [
                add_pth(txt_root, "eager_train_step_time_list.txt"),
                add_pth(txt_root, "graph_train_step_time_list.txt"),
            ],
            "names": ["eager_step_time", "graph_step_time"],
            "xlabel": "iters",
            "ylabel": "time(s)",
            "ylim": [0, 1],
        }
    )
    draw_and_save(
        {
            "title": "compare_train_epoch_time",
            "save_path": add_pth(save_root, "compare_train_epoch_time.png"),
            "txts": [
                add_pth(txt_root, "eager_train_epoch_time_list.txt"),
                add_pth(txt_root, "graph_train_epoch_time_list.txt"),
            ],
            "names": ["eager_epoch_time", "graph_epoch_time"],
            "xlabel": "epochs",
            "ylabel": "time(s)",
        }
    )
    draw_and_save(
        {
            "title": "compare_eval_epoch_time",
            "save_path": add_pth(save_root, "compare_eval_epoch_time.png"),
            "txts": [
                add_pth(txt_root, "eager_eval_epoch_time_list.txt"),
                add_pth(txt_root, "graph_eval_epoch_time_list.txt"),
            ],
            "names": ["eager_eval_time", "graph_eval_time"],
            "xlabel": "epochs",
            "ylabel": "time(s)",
        }
    )
    draw_and_save(
        {
            "title": "compare_abs_loss",
            "save_path": add_pth(save_root, "compare_abs_loss.png"),
            "txts": [
                add_pth(txt_root, "eager_losses.txt"),
                add_pth(txt_root, "graph_losses.txt"),
            ],
            "names": ["eager_loss", "graph_loss"],
            "xlabel": "iters",
            "ylabel": "abs_loss",
            "do_abs_minus": True,
        }
    )
    draw_and_save(
        {
            "title": "compare_abs_model_param",
            "save_path": add_pth(save_root, "compare_abs_model_param.png"),
            "txts": [add_pth(txt_root, "eager_graph_model_diff_list.txt")],
            "names": ["model_param_abs_diff"],
            "xlabel": "iters",
            "ylabel": "abs_diff",
        }
    )
