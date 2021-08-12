import os
from typing import Dict, Iterable, List

import numpy as np


class Reporter:

    @staticmethod
    def calc_corr(a: List, b: List) -> float:
        def square(lst): return list(map(lambda x: x**2, lst))

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
            loss_metric1: List, loss_metric2: List,
            train_acc_metric1: List, train_acc_metric2: List,
            val_acc_metric1: List, val_acc_metric2: List,
            train_time_metric1: List, train_time_metric2: List,
            val_time_metric1: List, val_time_metric2: List
    ) -> None:

        abs_loss_diff = abs(np.array(loss_metric1) - np.array(loss_metric2))
        loss_corr = cls.calc_corr(loss_metric1, loss_metric2)

        train_acc_corr = cls.calc_corr(train_acc_metric1, train_acc_metric2)
        val_acc_corr = cls.calc_corr(val_acc_metric1, val_acc_metric2)

        train_time_compare = np.divide(
            train_time_metric1, train_time_metric2).mean()
        val_time_compare = np.divide(val_time_metric1, val_time_metric2).mean()

        # Write to reporter
        save_path = os.path.join("check", 'check_report.txt')
        writer = open(save_path, "w")
        writer.write("Check Report\n")
        writer.write("Model: {}\n".format(model_name))
        writer.write("Check Results Between Eager Model and Graph Model\n")
        writer.write("=================================================\n")
        writer.write("Loss Correlation: %.4f\n\n" % loss_corr)
        writer.write("Max Loss Difference: %.4f\n" % abs_loss_diff.max())
        writer.write("Min Loss Difference: %.4f\n" % abs_loss_diff.min())
        writer.write("Loss Difference Range: (%.4f, %.4f)\n\n" %
                     (abs_loss_diff.min(), abs_loss_diff.max()))
        writer.write("Train Accuracy Correlation: %.4f\n\n" % train_acc_corr)
        writer.write("Val Accuracy Correlation: %.4f\n\n" % val_acc_corr)
        writer.write("Train Time Compare: %.4f (Eager) : %.4f (Graph)\n\n" % (
            train_time_compare, 1.0))
        writer.write("Val Time Compare: %.4f (Eager) : %.4f (Graph)\n" %
                     (val_time_compare, 1.0))
        writer.close()
        print("Report saved to: ", save_path)
