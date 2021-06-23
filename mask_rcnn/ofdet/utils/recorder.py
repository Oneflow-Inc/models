import time
import datetime


class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        self.last_split = self.start_time

    def split(self):
        now = time.perf_counter()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.perf_counter()

    def duration(self):
        return self.stop_time - self.start_time


class InferenceRecorder(object):
    def __init__(self, iteration_num, evaluator=None):
        self.iteration_num = iteration_num
        self.num_warmp = min(5, iteration_num - 1)
        self.timer = Timer()
        self.evaluator = evaluator
        self.pred_list = []
        self.total_compute_time = 0
        self.last_logged = None

    def record_cb(self, step):
        def callback(outputs):
            if step == self.num_warmp:
                self.timer.reset()
                self.total_compute_time = 0
            assert isinstance(outputs, dict)
            img_ids = outputs["img_ids"]
            img_ids = list(img_ids.numpy(0))

            for idx in range(len(img_ids)):
                pred_dict = {
                    "img_id": img_ids[idx],
                    "boxes": outputs["boxes"][idx].numpy(0),
                    "scores": outputs["scores"][idx].numpy(0),
                    "labels": outputs["labels"][idx].numpy(0),
                    "img_size": outputs["img_size"][idx].numpy(0),
                    "orig_img_size": outputs["orig_img_size"].numpy(0)[idx],
                }
                if "masks" in outputs.keys():
                    pred_dict["masks"] = outputs["masks"][idx].numpy(0)
                self.pred_list.append(pred_dict)

            self.total_compute_time += self.timer.split()
            iters_after_start = step + 1 - self.num_warmp * int(step >= self.num_warmp)
            seconds_per_img = self.total_compute_time / iters_after_start
            if step >= self.num_warmp * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - self.timer.start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (self.iteration_num - step - 1)))
                current_time = time.time()
                if self.last_logged is None or current_time - self.last_logged > 5:
                    print("Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        step + 1, self.iteration_num, seconds_per_img, str(eta)))
                    self.last_logged = current_time
            if ((step + 1) % self.iteration_num == 0):
                self.timer.stop()
                print("whole time: {}.".format(str(datetime.timedelta(seconds=int(self.timer.duration())))))
                self.evaluator.process(self.pred_list)
                self.pred_list.clear()

        return callback
