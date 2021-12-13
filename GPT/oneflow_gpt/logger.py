import time
import numpy as np
import oneflow as flow


class Logger(object):
    def __init__(self, rank):
        self.rank = rank
        self.step = 0
        self.metrics = dict()

    def register_metric(
        self, metric_name, metric, print_format=None, reset_after_print=False
    ):
        if metric_name in self.metrics:
            raise ValueError(f"{metric_name} is already registered")

        self.metrics[metric_name] = {
            "metric": metric,
            "print_format": print_format or (metric_name + ": {}"),
            "reset_after_print": reset_after_print,
        }

    def metric(self, mkey):
        if mkey not in self.metrics:
            return None

        return self.metrics[mkey]["metric"]

    def meter(self, mkey, *args, **kwargs):
        if mkey not in self.metrics:
            raise ValueError(f"{mkey} is not registered")

        self.metrics[mkey]["metric"].record(*args, **kwargs)

    def print_metrics(self, ranks=None):
        fields = []
        for m in self.metrics.values():
            metirc = m["metric"]
            fields.append(metirc.get_format_str(m["print_format"]))
            if m["reset_after_print"]:
                metirc.reset()

        print_ranks(ranks, "[rank:{}] {}".format(self.rank, ", ".join(fields)))


class IterationMetric(object):
    def __init__(self):
        self.val = 0

    def record(self, val):
        self.val = val

    def get_format_str(self, pattern):
        return pattern.format(self.val)


class LossMetric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.numel = 0
        self.loss_sum = None

    def record(self, loss):
        if isinstance(loss, flow.Tensor):
            self.numel += loss.shape.numel()
            loss = loss.sum()
            if self.loss_sum is None:
                self.loss_sum = flow.zeros_like(loss)
            self.loss_sum += loss
        elif isinstance(loss, np.ndarray):
            self.numel += loss.size
            loss = loss.sum()
            if self.loss_sum is None:
                self.loss_sum = flow.zeros_like(loss)
            self.loss_sum += loss
        elif isinstance(loss, float):
            self.numel += 1
            if self.loss_sum is None:
                self.loss_sum = 0.0
            self.loss_sum += loss
        elif isinstance(loss, int):
            self.numel += 1
            if self.loss_sum is None:
                self.loss_sum = 0
            self.loss_sum += loss
        else:
            raise TypeError(f"invalid loss type: {type(loss)}")

    def get_avg_loss(self):
        if isinstance(self.loss_sum, flow.Tensor):
            # NOTE(zwx): sync happen here
            loss_sum = self.loss_sum.numpy().item()
        elif isinstance(self.loss_sum, np.ndarray):
            loss_sum = self.loss_sum.item()
        else:
            loss_sum = self.loss_sum

        if self.numel == 0:
            return 0
        else:
            return loss_sum / self.numel

    def get_format_str(self, pattern):
        loss = self.get_avg_loss()
        return pattern.format(loss)


class AccumulationMetric(object):
    def __init__(self):
        self.acc = 0

    def record(self, n):
        self.acc += n

    def get_format_str(self, pattern):
        return pattern.format(self.acc)


class ThroughputMetric(object):
    def __init__(self):
        self.n = 0
        self.ets = None
        self.bts = None
        self.reset()

    def reset(self):
        self.n = 0
        if self.ets is None:
            self.bts = time.perf_counter()
        else:
            self.bts = self.ets
        self.ets = None

    def record(self, n):
        self.n += n

    def get_format_str(self, pattern):
        self.ets = time.perf_counter()
        assert self.ets > self.bts, f"{self.ets} > {self.bts}"
        throughput = self.n / (self.ets - self.bts)
        return pattern.format(throughput)


def print_rank_0(*args, **kwargs):
    if flow.env.get_rank() == 0:
        print(*args, **kwargs)


def print_rank_last(*args, **kwargs):
    if flow.env.get_rank() == flow.env.get_world_size() - 1:
        print(*args, **kwargs)


def print_ranks(ranks, *args, **kwargs):
    rank = flow.env.get_rank()
    if ranks is None:
        ranks = range(flow.env.get_world_size())

    if rank in ranks:
        print(*args, **kwargs)
