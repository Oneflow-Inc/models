from collections import OrderedDict
import time
import os


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


class Metric(object):
    def __init__(
        self,
        desc="train",
        print_steps=-1,
        batch_size=256,
        keys=[],
        nvidia_smi_report_step=10,
    ):
        r"""accumulate and calculate metric

        Args:
            desc: `str` general description of the metric to show
            print_steps: `Int` print metrics every nth steps
            batch_size: `Int` batch size per step
            keys: keys in callback outputs
        Returns:
        """
        self.desc = desc
        self.print_steps = print_steps
        assert batch_size > 0
        self.batch_size = batch_size
        self.nvidia_smi_report_step = nvidia_smi_report_step

        assert isinstance(keys, (list, tuple))
        self.keys = keys
        self.metric_dict = OrderedDict()
        self.metric_dict["step"] = 0

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        for key in self.keys:
            self.metric_dict[key] = 0.0
            self.metric_dict["n_" + key] = 0.0
        self.metric_dict["throughput"] = 0.0
        self.num_samples = 0.0

    def update_and_save(self, key, value, step, **kwargs):
        self.metric_dict[key] = value
        self.metric_dict.pop("n_" + key, None)

    def metric_cb(self, step=0, **kwargs):
        def callback(outputs):
            if step == 0:
                self._clear()

            if step == self.nvidia_smi_report_step:
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
                os.system(cmd)

            for key in self.keys:
                self.metric_dict[key] += outputs[key].sum()
                self.metric_dict["n_" + key] += outputs[key].size

            self.num_samples += self.batch_size

            if (step + 1) % self.print_steps == 0:
                self.metric_dict["step"] = step
                for k, v in kwargs.items():
                    self.metric_dict[k] = v
                throughput = self.num_samples / self.timer.split()
                self.update_and_save("throughput", throughput, step)
                for key in self.keys:
                    value = self.metric_dict[key] / self.metric_dict["n_" + key]
                    self.update_and_save(key, value, step, **kwargs)
                print(
                    ", ".join(
                        ("{}: {}" if type(v) is int else "{}: {:.3f}").format(k, v)
                        for k, v in self.metric_dict.items()
                    ),
                    time.time(),
                )
                self._clear()

        return callback
