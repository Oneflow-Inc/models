import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict


def draw_result(
    save_dir: str, name, xlabel: str, ylabel: str, data: Dict[str, np.ndarray],
) -> None:
    # Setup matplotlib
    plt.rcParams["figure.dpi"] = 100
    plt.clf()
    for data_name, values in data.items():
        axis = np.arange(1, len(values) + 1)
        # Draw Line Chart
        plt.plot(axis, values, "-", linewidth=1.5, label=data_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best", frameon=True, fontsize=8)
    plt.savefig(os.path.join(save_dir, name + ".png"))


if __name__ == "__main__":
    # with open("./temp/bert_graph_loss.txt", "r") as f:
    # eager_total_loss = [float(line) for line in f.readlines()]
    with open("./loss_txt/bert_graph_sgd_loss.txt", "r") as f:
        graph_total_loss = [float(line) for line in f.readlines()]
    with open(
        "../../OneFlow-Benchmark/LanguageModeling/BERT/loss_info_sgd.txt", "r"
    ) as f:
        lazy_total_loss = [float(line) for line in f.readlines()]

    draw_result(
        "loss_txt",
        "lazy_graph_sgd_loss",
        "steps",
        "loss",
        {
            # "eager": eager_total_loss,
            "lazy": lazy_total_loss,
            "graph": graph_total_loss,
        },
    )
    # with open("./temp/eager_lml_loss.txt", 'r') as f:
    #     eager_total_loss = [float(line) for line in f.readlines()]
    # with open("../../OneFlow-Benchmark/LanguageModeling/BERT/temp1/lazy_mlm_loss.txt", 'r') as f:
    #     lazy_total_loss = [float(line) for line in f.readlines()]

    # draw_result("loss_curve", "mlm_loss", "steps", "loss", {"eager": eager_total_loss, "lazy": lazy_total_loss})
