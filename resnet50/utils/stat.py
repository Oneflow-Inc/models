import sys
import signal
import subprocess
import threading


class CudaUtilMemStat:
    def __init__(
        self, stat_file_path, stat_util=True, stat_mem=True, only_ordinal=None
    ):
        self.stat_file = open(stat_file_path, "wt")
        self.stat_util = stat_util
        self.stat_mem = stat_mem
        self.only_ordinal = only_ordinal
        self._write_titles()

    def __del__(self):
        self.stat_file.close()

    def _write_titles(self):
        proc = subprocess.Popen(
            ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if stderr != b"":
            raise RuntimeError(stderr)

        gpus = []
        lines = stdout.decode("utf-8").split("\n")
        for line in lines:
            if line.strip() == "":
                continue
            gpus.append(line.split(":")[0])

        util_titles, mem_titles = [], []
        for gpu in gpus:
            if self.stat_util:
                util_titles.append(gpu + " utilization")
            if self.stat_mem:
                mem_titles.append(gpu + " memory used")

        if self.only_ordinal is None:
            self.stat_file.write(",".join(util_titles + mem_titles) + "\n")
        else:
            titles = (
                util_titles[self.only_ordinal] + "," + mem_titles[self.only_ordinal]
            )
            self.stat_file.write(titles + "\n")
            self.stat_file.flush()

    def stat(self):
        # command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if stderr != b"":
            raise RuntimeError(stderr)

        lines = stdout.decode("utf-8").split("\n")
        util_vals = []
        mem_vals = []
        for line in lines[1:]:
            if line.strip() == "":
                continue
            util_mem = line.split(",")
            assert len(util_mem) == 2, lines
            util = util_mem[0].strip()
            mem = util_mem[1].strip()
            assert "%" in util
            assert "MiB" in mem
            util = util.split(" ")
            mem = mem.split(" ")
            assert len(util) == 2
            assert len(mem) == 2
            util_val = util[0].strip()
            mem_val = mem[0].strip()
            if self.stat_util:
                util_vals.append(util_val)
            if self.stat_mem:
                mem_vals.append(mem_val)

        if self.only_ordinal is None:
            self.stat_file.write(",".join(util_vals + mem_vals) + "\n")
        else:
            vals = util_vals[self.only_ordinal] + "," + mem_vals[self.only_ordinal]
            self.stat_file.write(vals + "\n")
            self.stat_file.flush()

    def start(self, interval):
        stop = threading.Event()
        stat_thrd = StatThread(self.stat, interval, stop)
        stat_thrd.start()

        def close(signum, frame):
            print("Closing...")
            stop.set()

        signal.signal(signal.SIGTERM, close)
        signal.signal(signal.SIGINT, close)
        print("Start stat")
        print("Print Ctrl+C to stop")
        stop.wait()


class StatThread(threading.Thread):
    def __init__(self, handler, interval, stop_event):
        super().__init__()
        self.handler = handler
        self.interval = interval
        self.stopped = stop_event
        self.count = 0

    def run(self):
        while not self.stopped.wait(self.interval):
            print(f"{self.count} th run stat")
            self.handler()
            self.count += 1


if __name__ == "__main__":
    stat_file_path = sys.argv[1] if len(sys.argv) > 1 else "gpu_stat.log"
    interval = sys.argv[2] if len(sys.argv) > 2 else 1
    stat = CudaUtilMemStat(stat_file_path)
    stat.start(interval)
