import numpy as np
import pandas as pd
import statistics
import time
from collections import OrderedDict
from datetime import datetime


def transpose_metrics(metrics):
    transposed = metrics.pivot_table(
        values="value", columns=["legend"], aggfunc="mean", dropna=False
    )
    assert metrics["iter"].unique().size == 1, "can only transpose metrics in one iter"
    transposed["iter"] = metrics["iter"].unique()
    return transposed


def print_metrics(m):
    to_print_with_order = [
        "iter",
        "rank",
        "loss_rpn_box_reg",
        "loss_objectness",
        "loss_box_reg",
        "loss_classifier",
        "loss_mask",
        "mask_rcnn/false_positive",
        "mask_rcnn/false_negative",
        "train_step",
        "lr",
        "lr2",
        "elapsed_time",
    ]
    to_print_with_order = [l for l in to_print_with_order if l in m]
    print(m[to_print_with_order].to_string(index=False, float_format='%11.6f'))


def add_metrics(metrics_df, iter=None, **kwargs):
    assert iter is not None
    for key, val in kwargs.items():
        if key == "outputs":
            if isinstance(val, list):
                dfs = []
                for rank, val in enumerate(val, 0):
                    for legend, value in val.items():
                        dfs.append(
                            pd.DataFrame(
                                {
                                    "iter": iter,
                                    "rank": rank,
                                    "legend": legend,
                                    "value": value.item(),
                                },
                                index=[0],
                            )
                        )
            elif isinstance(val, dict):
                dfs = []
                for legend, value in val.items():
                    dfs.append(
                        pd.DataFrame(
                            {
                                "iter": iter,
                                "legend": legend,
                                "value": value.item()
                            },
                            index=[0],
                        )
                    )
            else:
                raise ValueError("not supported")
            metrics_df = pd.concat([metrics_df] + dfs, axis=0, sort=False)
        elif key == "elapsed_time":
            metrics_df = pd.concat(
                [
                    metrics_df,
                    pd.DataFrame(
                        {"iter": iter, "legend": key, "value": val, "rank": 0}, index=[0]
                    ),
                ],
                axis=0,
                sort=False,
            )
        elif key != "outputs":
            metrics_df = pd.concat(
                [
                    metrics_df,
                    pd.DataFrame({"iter": iter, "legend": key, "value": val}, index=[0]),
                ],
                axis=0,
                sort=False,
            )
        else:
            raise ValueError("not supported")
    return metrics_df


class IterProcessor(object):
    def __init__(self, start_iter, cfg):
        self.start_time = time.perf_counter()
        self.elapsed_times = []
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.save_metrics_period = cfg.SOLVER.METRICS_PERIOD
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.img_per_batch = cfg.SOLVER.IMS_PER_BATCH
        self.ngpus = cfg.ENV.NUM_GPUS
        self.image_dir = (cfg.DATASETS.IMAGE_DIR_TRAIN, )
        self.metrics = pd.DataFrame(
            {"iter": start_iter, "legend": "cfg", "note": str(cfg)}, index=[0]
        )
        self.start_iter = start_iter
        self.collect_accuracy_metrics = cfg.MODEL.COLLECT_ACCURACY_METRICS

    def process_one_rank(self, outputs_dict):
        assert isinstance(outputs_dict, dict)
        outputs_dict.pop("image_id")
        for key, value in outputs_dict.items():
            outputs_dict[key] = value.numpy_list()[0]
        if self.collect_accuracy_metrics:
            mask_incorrect = outputs_dict.pop("mask_rcnn/mask_incorrect").astype(np.bool)
            gt_masks_bool = outputs_dict.pop("mask_rcnn/gt_masks_bool").astype(np.bool)
            num_positive = outputs_dict.pop("mask_rcnn/num_positive").item()
            false_positive = (mask_incorrect & ~gt_masks_bool).sum() / max(
                gt_masks_bool.size - num_positive, 1.0
            )
            false_negative = (mask_incorrect & gt_masks_bool).sum() / max(num_positive, 1.0)
            outputs_dict["mask_rcnn/false_positive"] = false_positive
            outputs_dict["mask_rcnn/false_negative"] = false_negative

    def outputs_postprocess(self, outputs):
        if isinstance(outputs, (list, tuple)):
            for outputs_per_rank in outputs:
                self.process_one_rank(outputs_per_rank)
        elif isinstance(outputs, dict):
            self.process_one_rank(outputs)
        else:
            raise ValueError("outputs has error type")

    def step(self, iteration, verbose=True):
        def callback(outputs):
            now_time = time.perf_counter()
            elapsed_time = now_time - self.start_time
            self.elapsed_times.append(elapsed_time)
            self.outputs_postprocess(outputs)
            metrics_df = pd.DataFrame()
            metrics_df = add_metrics(metrics_df, iter=iteration, elapsed_time=elapsed_time)
            metrics_df = add_metrics(metrics_df, iter=iteration, outputs=outputs)
            rank_size = (
                metrics_df["rank"].dropna().unique().size if "rank" in metrics_df else 0
            )
            if verbose and rank_size > 1:
                for rank_i in range(rank_size):
                    transposed = transpose_metrics(metrics_df[metrics_df["rank"] == rank_i])
                    transposed["rank"] = rank_i
                    print_metrics(transposed)
            else:
                transposed = transpose_metrics(metrics_df)
                print_metrics(transposed)

            self.metrics = pd.concat([self.metrics, metrics_df], axis=0, sort=False)

            if self.save_metrics_period > 0 and (
                iteration % self.save_metrics_period == 0 or iteration == self.max_iter
            ):
                npy_file_name = "loss-{}-{}-batch_size-{}-gpu-{}-image_dir-{}-{}.csv".format(
                    self.start_iter,
                    iteration,
                    self.img_per_batch,
                    self.ngpus,
                    self.image_dir,
                    str(datetime.now().strftime("%Y-%m-%d--%H-%M-%S")),
                )
                npy_file_name = npy_file_name.replace("/", "-")
                self.metrics.to_csv(npy_file_name, index=False)
                print("saved: {}".format(npy_file_name))

            self.start_time = time.perf_counter()

            if iteration == self.max_iter:
                print(
                    "median of elapsed time per batch:",
                    statistics.median(self.elapsed_times)
                )

        return callback


class _Metrics(object):
    def __init__(self):
        self.metrics = OrderedDict()

    def get_metrics(self):
        return self.metrics

    def put_metrics(self, metrics):
        self.metrics.update(metrics)
        return self.metrics

    def clear_metrics(self):
        """
        In mirror mode, it is recommended to clear the dict when one rank is
        built.
        """
        self.metrics.clear()


Metrics = _Metrics()
