import time
from datetime import datetime
import numpy as np
import oneflow as flow

_GLOBAL_LOGGER = None


def get_logger(rank, print_ranks):
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = Logger(rank, print_ranks)

    return _GLOBAL_LOGGER


class Logger(object):
    def __init__(self, rank, print_ranks):
        self.rank = rank
        self.print_ranks = print_ranks
        # self.verbose = verbose
        # self.b = backend
        self.step = 0
        self.m = dict()

    def register_metric(
        self, metric_key, meter, print_format=None, reset_after_print=False
    ):
        assert metric_key not in self.m
        self.m[metric_key] = {
            "meter": meter,
            "print_format": print_format or (metric_key + ": {}"),
            "reset_after_print": reset_after_print,
        }

    def metric(self, mkey):
        if mkey not in self.m:
            return None

        return self.m[mkey]["meter"]

    def meter(self, mkey, *args):
        assert mkey in self.m
        self.m[mkey]["meter"].record(*args)

    def print_metrics(self, print_ranks=None):
        fields = []
        for m in self.m.values():
            meter = m["meter"]
            print_format = m["print_format"]
            result = meter.get()
            if isinstance(result, (list, tuple)):
                field = print_format.format(*result)
            else:
                field = print_format.format(result)
            fields.append(field)
            if m["reset_after_print"]:
                meter.reset()

        do_print = self.rank in (print_ranks or self.print_ranks)
        if do_print:
            print(
                "[rank:{}] {}".format(self.rank, ", ".join(fields)),
                datetime.now().strftime("| %Y-%m-%d %H:%M:%S.%f")[:-3],
            )

    def print(self, *args, print_ranks=None):
        do_print = self.rank in (print_ranks or self.print_ranks)
        if do_print:
            print(*args)


class IterationMeter(object):
    def __init__(self):
        self.val = 0

    def record(self, val):
        self.val = val

    def get(self):
        return self.val


def _zeros_by_val(val):
    ret = 0
    if isinstance(val, flow.Tensor):
        ret = flow.zeros_like(val)
    elif isinstance(val, np.ndarray):
        ret = np.zeros_like(val)
    elif isinstance(val, int):
        ret = 0
    elif isinstance(val, float):
        ret = 0.0
    else:
        raise ValueError
    return ret


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = None
        self.n = 0

    def record(self, val, n=1):
        self.n += n

        if self.sum is None:
            self.sum = _zeros_by_val(val)

        if n == 1:
            self.sum += val
        else:
            self.sum += val * n

    def get(self):
        if self.n == 0:
            return 0

        avg = self.sum / self.n
        if isinstance(avg, flow.Tensor):
            # NOTE(zwx): sync happen here
            return avg.numpy().item()
        elif isinstance(avg, np.ndarray):
            return avg.item()
        else:
            return avg


class RunningMeter(object):
    pass


class ProgressMeter(object):
    def __init__(self, total):
        self.cur = 0
        self.total = total

    def record(self, cur, total=None):
        self.cur = cur
        if total is not None:
            self.total = total

    def get(self):
        return self.cur, self.total


class TimeMeter(object):
    def __init__(self, return_timestamp=False):
        self.return_timestamp = return_timestamp
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

    def get(self):
        self.ets = time.perf_counter()
        assert self.ets > self.bts, f"{self.ets} > {self.bts}"
        throughput = self.n / (self.ets - self.bts)
        if self.return_timestamp:
            return throughput, self.ets
        else:
            return throughput
