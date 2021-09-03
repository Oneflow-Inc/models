import oneflow as flow
import argparse
import numpy as np
import os
import time
from tqdm import tqdm

import sys

sys.path.append(".")
from models.resnet50 import resnet50
from utils.ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
    parser = argparse.ArgumentParser("flags for train resnet50")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path", type=str, default="./ofrecord/", help="dataset path"
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=4, help="val batch size")
    parser.add_argument(
        "--results", type=str, default="./results", help="tensorboard file path"
    )
    parser.add_argument("--tag", type=str, default="default", help="tag of experiment")
    parser.add_argument(
        "--print_interval", type=int, default=10, help="print info frequency"
    )
    return parser.parse_args()


def setup(args):
    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=9469,
        batch_size=args.train_batch_size,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=3925,
        batch_size=args.val_batch_size,
    )

    criterion = flow.nn.CrossEntropyLoss()

    # model setup
    eager_model = resnet50()
    graph_model = resnet50()
    graph_model.load_state_dict(eager_model.state_dict())

    eager_model.to("cuda")
    graph_model.to("cuda")
    # optimizer setup
    eager_optimizer = flow.optim.SGD(
        eager_model.parameters(), lr=args.learning_rate, momentum=args.mom
    )
    graph_optimizer = flow.optim.SGD(
        graph_model.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    # criterion setup
    criterion = flow.nn.CrossEntropyLoss()
    criterion = criterion.to("cuda")

    class ModelTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.graph_model = graph_model
            self.criterion = criterion
            self.add_optimizer(graph_optimizer)

        def build(self, image, label):
            logits = self.graph_model(image)
            loss = self.criterion(logits, label)
            loss.backward()
            return loss

    class ModelEvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.graph_model = graph_model

        def build(self, image):
            with flow.no_grad():
                logits = self.graph_model(image)
                predictions = logits.softmax()
            return predictions

    model_train_graph = ModelTrainGraph()
    model_eval_graph = ModelEvalGraph()

    dic = {
        "train_dataloader": train_data_loader,
        "val_dataloader": val_data_loader,
        "eager": [eager_model, eager_optimizer, criterion],
        "graph": [graph_model, model_train_graph, model_eval_graph],
    }

    return dic


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.graph_losses = []
        self.eager_losses = []

        self.graph_acc = []
        self.eager_acc = []

        self.graph_train_step_time_list = []
        self.eager_train_step_time_list = []

        self.graph_train_epoch_time_list = []
        self.eager_train_epoch_time_list = []

        self.graph_eval_epoch_time_list = []
        self.eager_eval_epoch_time_list = []

        self.eager_graph_model_diff_list = []

        self.graph_train_total_time = 0.0
        self.eager_train_total_time = 0.0

        self.graph_eval_total_time = 0.0
        self.eager_val_total_time = 0.0

        self.args = args

    def compare_eager_graph(self, compare_dic):

        train_data_loader = compare_dic["train_dataloader"]
        val_data_loader = compare_dic["val_dataloader"]
        eager_model, eager_optimizer, criterion = compare_dic["eager"]
        graph_model, model_train_graph, model_eval_graph = compare_dic["graph"]

        all_samples = len(val_data_loader) * self.args.val_batch_size
        print_interval = self.args.print_interval

        print("start training")
        for epoch in range(self.args.epochs):
            # train
            eager_model.train()
            graph_model.train()
            start_training_time = time.time()
            total_graph_iter_time, total_eager_iter_time = 0, 0

            for b in range(len(train_data_loader)):
                image, label = train_data_loader()
                image = image.to("cuda")
                label = label.to("cuda")

                # oneflow graph train
                graph_iter_start_time = time.time()
                graph_loss = model_train_graph(image, label)
                graph_loss.numpy()  # for synchronize CPU and GPU, get accurate running time
                graph_iter_end_time = time.time()

                # oneflow eager train
                eager_iter_start_time = time.time()
                logits = eager_model(image)
                eager_loss = criterion(logits, label)
                eager_loss.backward()
                eager_optimizer.step()
                eager_optimizer.zero_grad()
                eager_loss.numpy()  # for synchronize CPU and GPU, get accurate running time
                eager_iter_end_time = time.time()

                model_param_diff = compare_model_params(eager_model, model_train_graph)
                self.eager_graph_model_diff_list.append(model_param_diff)

                # get time
                graph_iter_time = graph_iter_end_time - graph_iter_start_time
                eager_iter_time = eager_iter_end_time - eager_iter_start_time
                total_graph_iter_time += graph_iter_time
                total_eager_iter_time += eager_iter_time

                if b % print_interval == 0:
                    gl, el = graph_loss.numpy(), eager_loss.numpy()
                    print(
                        "epoch {} train iter {} ; graph loss {} eager loss {};  graph train time: {}  eager train time {}".format(
                            epoch, b, gl, el, graph_iter_time, eager_iter_time
                        )
                    )
                    self.graph_losses.append(gl)
                    self.graph_train_step_time_list.append(graph_iter_time)
                    self.eager_losses.append(el)
                    self.eager_train_step_time_list.append(eager_iter_time)

            end_training_time = time.time()
            self.graph_train_epoch_time_list.append(
                end_training_time - start_training_time - total_eager_iter_time
            )
            self.eager_train_epoch_time_list.append(
                end_training_time - start_training_time - total_graph_iter_time
            )
            print("epoch %d train done, start validation" % epoch)

            # validate
            eager_model.eval()
            graph_model.eval()
            graph_correct, eager_correct = 0.0, 0.0
            eval_start_time = time.time()
            total_graph_infer_time, total_eager_infer_time = 0, 0
            for b in tqdm(range(len(val_data_loader))):
                image, label = val_data_loader()
                image = image.to("cuda")

                # graph val
                graph_infer_time = time.time()
                predictions = model_eval_graph(image)
                graph_preds = predictions.numpy()
                graph_clsidxs = np.argmax(graph_preds, axis=1)
                total_graph_infer_time += time.time() - graph_infer_time

                # eager val
                eager_infer_time = time.time()
                with flow.no_grad():
                    logits = eager_model(image)
                    predictions = logits.softmax()
                eager_preds = predictions.numpy()
                eager_clsidxs = np.argmax(eager_preds, axis=1)
                total_eager_infer_time += time.time() - eager_infer_time

                label_nd = label.numpy()
                for i in range(self.args.val_batch_size):
                    if graph_clsidxs[i] == label_nd[i]:
                        graph_correct += 1
                    if eager_clsidxs[i] == label_nd[i]:
                        eager_correct += 1
            eval_end_time = time.time()
            self.graph_eval_epoch_time_list.append(
                eval_end_time - eval_start_time - total_eager_infer_time
            )
            self.eager_eval_epoch_time_list.append(
                eval_end_time - eval_start_time - total_graph_infer_time
            )
            graph_top1_acc, eager_top1_acc = (
                graph_correct / all_samples,
                eager_correct / all_samples,
            )
            self.graph_acc.append(graph_top1_acc)
            self.eager_acc.append(eager_top1_acc)
            print(
                "epoch %d, graph top1 val acc: %f, eager top1 val acc: %f"
                % (epoch, graph_top1_acc, eager_top1_acc)
            )

    def save_report(self,):
        print("***** Save Report *****")
        # folder setup
        report_path = os.path.join(self.args.results)
        os.makedirs(report_path, exist_ok=True)

        # calculate absolute loss difference
        abs_loss_diff = abs(np.array(self.eager_losses) - np.array(self.graph_losses))

        # calculate losses linear correlation
        loss_corr = calc_corr(self.eager_losses, self.graph_losses)

        # calculate accuracy linear correlation
        acc_corr = calc_corr(self.eager_acc, self.graph_acc)

        # training time compare
        train_time_compare = time_compare(
            self.graph_train_epoch_time_list, self.eager_train_epoch_time_list
        )

        # validate time compare
        val_time_compare = time_compare(
            self.graph_eval_epoch_time_list, self.eager_eval_epoch_time_list
        )

        # eager graph model diff compare
        model_diff_compare = np.array(self.eager_graph_model_diff_list)

        # save report
        save_path = os.path.join(report_path, "check_report.txt")
        writer = open(save_path, "w")
        writer.write("Check Report\n")
        writer.write("Model: Resnet50\n")
        writer.write("Check Results Between Eager Model and Graph Model\n")
        writer.write("=================================================\n")
        writer.write("Loss Correlation: %.4f\n\n" % loss_corr)
        writer.write("Max Loss Difference: %.4f\n" % abs_loss_diff.max())
        writer.write("Min Loss Difference: %.4f\n" % abs_loss_diff.min())
        writer.write(
            "Loss Difference Range: (%.4f, %.4f)\n\n"
            % (abs_loss_diff.min(), abs_loss_diff.max())
        )
        writer.write(
            "Model Param Difference Range: (%.4f, %.4f)\n\n"
            % (model_diff_compare.min(), model_diff_compare.max())
        )
        writer.write("Accuracy Correlation: %.4f\n\n" % acc_corr)
        writer.write(
            "Train Time Compare: %.4f (Eager) : %.4f (Graph)\n\n"
            % (1.0, train_time_compare)
        )
        writer.write(
            "Val Time Compare: %.4f (Eager) : %.4f (Graph)" % (1.0, val_time_compare)
        )
        writer.close()
        print("Report saved to: ", save_path)

    def save_result(self,):
        # create folder
        training_results_path = os.path.join(self.args.results, self.args.tag)
        os.makedirs(training_results_path, exist_ok=True)
        print("***** Save Results *****")
        save_results(
            self.graph_losses, os.path.join(training_results_path, "graph_losses.txt")
        )
        save_results(
            self.eager_losses, os.path.join(training_results_path, "eager_losses.txt")
        )

        save_results(
            self.graph_acc, os.path.join(training_results_path, "graph_acc.txt")
        )
        save_results(
            self.eager_acc, os.path.join(training_results_path, "eager_acc.txt")
        )

        save_results(
            self.graph_train_step_time_list,
            os.path.join(training_results_path, "graph_train_step_time_list.txt"),
        )
        save_results(
            self.eager_train_step_time_list,
            os.path.join(training_results_path, "eager_train_step_time_list.txt"),
        )

        save_results(
            self.graph_train_epoch_time_list,
            os.path.join(training_results_path, "graph_train_epoch_time_list.txt"),
        )
        save_results(
            self.eager_train_epoch_time_list,
            os.path.join(training_results_path, "eager_train_epoch_time_list.txt"),
        )

        save_results(
            self.graph_eval_epoch_time_list,
            os.path.join(training_results_path, "graph_eval_epoch_time_list.txt"),
        )
        save_results(
            self.eager_eval_epoch_time_list,
            os.path.join(training_results_path, "eager_eval_epoch_time_list.txt"),
        )

        save_results(
            self.eager_graph_model_diff_list,
            os.path.join(training_results_path, "eager_graph_model_diff_list.txt"),
        )

        print("Results saved to: ", training_results_path)


def compare_model_params(eager_model, graph_model):
    num_params = len(eager_model.state_dict().keys())
    sum_diff = 0.0
    for key in eager_model.state_dict():
        mean_single_diff = (
            (
                eager_model.state_dict()[key]
                - graph_model.graph_model.state_dict()[key]._origin
            )
            .abs()
            .mean()
        )
        sum_diff += mean_single_diff
    mean_diff = float(sum_diff.numpy() / num_params)
    return mean_diff


def save_results(training_info, file_path):
    writer = open(file_path, "w")
    for info in training_info:
        writer.write("%f\n" % info)
    writer.close()


# report helpers
def square(lst):
    res = list(map(lambda x: x ** 2, lst))
    return res


# calculate correlation
def calc_corr(a, b):
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


def time_compare(a, b):
    return np.divide(a, b).mean()


if __name__ == "__main__":
    args = _parse_args()
    trainer = Trainer(args)
    compare_dic = setup(args)
    print("init done")
    trainer.compare_eager_graph(compare_dic)
    del compare_dic

    # save results
    trainer.save_result()
    trainer.save_report()
