import os
from typing import Dict, Iterable, List

from collections import defaultdict
import numpy as np
import glob

import matplotlib.pyplot as plt


class Reporter:
    @staticmethod
    def calc_corr(a: List, b: List) -> float:
        def square(lst):
            return list(map(lambda x: x ** 2, lst))

        E_a = np.mean(a)
        E_b = np.mean(b)
        E_ab = np.mean(list(map(lambda x: x[0] * x[1], zip(a, b))))

        cov_ab = E_ab - E_a * E_b

        D_a = np.mean(square(a)) - E_a ** 2
        D_b = np.mean(square(b)) - E_b ** 2

        σ_a = np.sqrt(D_a)
        σ_b = np.sqrt(D_b)

        corr_factor = cov_ab / (σ_a * σ_b)
        return corr_factor

    @classmethod
    def save_report(
        cls,
        model_name: str,
        save_dir: str,
        loss_metric1: List,
        loss_metric2: List,
        train_acc_metric1: List,
        train_acc_metric2: List,
        val_acc_metric1: List,
        val_acc_metric2: List,
        train_time_metric1: List,
        train_time_metric2: List,
        val_time_metric1: List,
        val_time_metric2: List,
    ) -> None:

        abs_loss_diff = abs(np.array(loss_metric1) - np.array(loss_metric2))
        loss_corr = cls.calc_corr(loss_metric1, loss_metric2)

        train_acc_corr = cls.calc_corr(train_acc_metric1, train_acc_metric2)
        val_acc_corr = cls.calc_corr(val_acc_metric1, val_acc_metric2)

        train_time_compare = np.divide(train_time_metric1, train_time_metric2).mean()
        val_time_compare = np.divide(val_time_metric1, val_time_metric2).mean()

        # Write to reporter
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "check_report.txt")
        writer = open(save_path, "w")
        writer.write("Check Report\n")
        writer.write("Model: {}\n".format(model_name))
        writer.write("Check Results Between Eager Model and Graph Model\n")
        writer.write("=================================================\n")
        writer.write("Loss Correlation: %.4f\n\n" % loss_corr)
        writer.write("Max Loss Difference: %.4f\n" % abs_loss_diff.max())
        writer.write("Min Loss Difference: %.4f\n" % abs_loss_diff.min())
        writer.write(
            "Loss Difference Range: (%.4f, %.4f)\n\n"
            % (abs_loss_diff.min(), abs_loss_diff.max())
        )
        writer.write("Train Accuracy Correlation: %.4f\n\n" % train_acc_corr)
        writer.write("Val Accuracy Correlation: %.4f\n\n" % val_acc_corr)
        writer.write(
            "Train Time Compare: %.4f (Eager) : %.4f (Graph)\n\n"
            % (train_time_compare, 1.0)
        )
        writer.write(
            "Val Time Compare: %.4f (Eager) : %.4f (Graph)\n" % (val_time_compare, 1.0)
        )
        writer.close()
        print("Report saved to: ", save_path)

    @staticmethod
    def write2file(save_info: List, file_path: str) -> None:
        writer = open(file_path, "w")
        for info in save_info:
            writer.write("%f\n" % info)
        writer.close()

    @classmethod
    def save_check_info(
        cls,
        save_dir: str,
        loss_metric: Dict,
        train_acc_metric: Dict,
        val_acc_metric: Dict,
        train_time_metric: Dict,
        val_time_metric: Dict,
    ) -> None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("**** Save Check info ****")

        for loss_name, loss_values in loss_metric.items():
            cls.write2file(loss_values, os.path.join(save_dir, loss_name + ".txt"))

        for train_acc_name, train_acc_values in train_acc_metric.items():
            cls.write2file(
                train_acc_values, os.path.join(save_dir, train_acc_name + ".txt")
            )

        for val_acc_name, val_acc_values in val_acc_metric.items():
            cls.write2file(
                val_acc_values, os.path.join(save_dir, val_acc_name + ".txt")
            )

        for train_time_name, train_time_values in train_time_metric.items():
            cls.write2file(
                train_time_values, os.path.join(save_dir, train_time_name + ".txt")
            )

        for val_time_name, val_time_values in val_time_metric.items():
            cls.write2file(
                val_time_values, os.path.join(save_dir, val_time_name + ".txt")
            )

        print("Check Results are saved to: {}".format(save_dir))

    @staticmethod
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

    @classmethod
    def draw_check_info(cls, save_dir: str) -> None:
        all_files = glob.glob(os.path.join(save_dir, "*.txt"))
        file_group = defaultdict(list)
        for file_path in all_files:
            if "check_report.txt" in file_path:
                continue
            group_name = os.path.basename(file_path).split(".")[0].split("_")[-1]
            file_group[group_name].append(file_path)

        for name, group_files in file_group.items():
            data_dict = {}
            for file_path in group_files:
                with open(file_path, "r") as f:
                    data = [float(line) for line in f.readlines()]
                data_dict[os.path.basename(file_path).split(".")[0]] = data
            xlabels = "steps" if "Acc" not in name else "epochs"
            ylabels = name
            cls.draw_result(save_dir, name, xlabels, ylabels, data_dict)
