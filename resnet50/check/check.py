import oneflow as flow
import argparse
import numpy as np
import os
import time
from tqdm import tqdm

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
    parser.add_argument("--results", type=str, default="./results", help="tensorboard file path")
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
                self.add_optimizer("sgd", graph_optimizer)

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

    dic = { "train_dataloader": train_data_loader,
            "val_dataloader": val_data_loader,
            "eager": [eager_model, eager_optimizer, criterion],
            "graph": [graph_model, model_train_graph, model_eval_graph]
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
                graph_loss.numpy()
                graph_iter_end_time = time.time()

                # oneflow eager train
                eager_iter_start_time = time.time()
                logits = eager_model(image)
                eager_loss = criterion(logits, label)
                eager_loss.backward()
                eager_optimizer.step()
                eager_optimizer.zero_grad()
                eager_loss.numpy()
                eager_iter_end_time = time.time()

                # print("=====graph fc param=====")
                # print(model_train_graph.graph_model.state_dict()['fc.bias']._origin[:50])
                # print("=====eager fc param=====")
                # print(eager_model.state_dict()['fc.bias'][:50])
                # import pdb
                # pdb.set_trace()

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
            self.graph_train_epoch_time_list.append(end_training_time - start_training_time- total_eager_iter_time)
            self.eager_train_epoch_time_list.append(end_training_time - start_training_time- total_graph_iter_time)
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
            self.graph_eval_epoch_time_list.append(eval_end_time - eval_start_time - total_eager_infer_time)
            self.eager_eval_epoch_time_list.append(eval_end_time - eval_start_time - total_graph_infer_time)
            graph_top1_acc, eager_top1_acc = graph_correct / all_samples, eager_correct / all_samples
            self.graph_acc.append(graph_top1_acc)
            self.eager_acc.append(eager_top1_acc)
            print("epoch %d, graph top1 val acc: %f, eager top1 val acc: %f" % (epoch, graph_top1_acc, eager_top1_acc))


def save_results(training_info, file_path):
    writer = open(file_path, "w")
    for info in training_info:
        writer.write("%f\n" % info)
    writer.close()

def save_result(trainer):
    # create folder
    training_results_path = os.path.join(args.results, args.tag)
    os.makedirs(training_results_path, exist_ok=True)
    print("***** Save Results *****")
    save_results(trainer.graph_losses, os.path.join(training_results_path, 'graph_losses.txt'))
    save_results(trainer.eager_losses, os.path.join(training_results_path, 'eager_losses.txt'))

    save_results(trainer.graph_acc, os.path.join(training_results_path, 'graph_acc.txt'))
    save_results(trainer.eager_acc, os.path.join(training_results_path, 'eager_acc.txt'))

    save_results(trainer.graph_train_step_time_list, os.path.join(training_results_path, 'graph_train_step_time_list.txt'))
    save_results(trainer.eager_train_step_time_list, os.path.join(training_results_path, 'eager_train_step_time_list.txt'))

    save_results(trainer.graph_train_epoch_time_list, os.path.join(training_results_path, 'graph_train_epoch_time_list.txt'))
    save_results(trainer.eager_train_epoch_time_list, os.path.join(training_results_path, 'eager_train_epoch_time_list.txt'))

    save_results(trainer.graph_eval_epoch_time_list, os.path.join(training_results_path, 'graph_eval_epoch_time_list.txt'))
    save_results(trainer.eager_eval_epoch_time_list, os.path.join(training_results_path, 'eager_eval_epoch_time_list.txt'))
    print("***** File Saved! *****")

if __name__ == "__main__":
    args = _parse_args()
    trainer = Trainer(args)
    compare_dic = setup(args)
    print("init done")
    trainer.compare_eager_graph(compare_dic)
    del compare_dic
    save_result(trainer)