import oneflow as flow
import argparse
import numpy as np
import os
import time
from tqdm import tqdm

from models.alexnet import alexnet
from utils.ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
    parser = argparse.ArgumentParser("flags for train alexnet")
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
        "--ofrecord_path", type=str, default="/data/imagenet/ofrecord/", help="dataset path"
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=32, help="val batch size")
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
        train_shuffle=False,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=3925,
        batch_size=args.val_batch_size,
    )

    criterion = flow.nn.CrossEntropyLoss()
    
    # model setup
    eager_model = alexnet()
    graph_model = alexnet()
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

    return train_data_loader, val_data_loader, eager_model, graph_model, eager_optimizer, graph_optimizer, criterion

# def build_graph(args, graph_model, criterion, optimizer):
#     criterion = criterion
#     optimizer = optimizer
#     class ModelTrainGraph(flow.nn.Graph):
#         def __init__(self):
#             super().__init__()
#             self.graph_model = graph_model
#             self.criterion = criterion
#             self.add_optimizer("sgd", optimizer)
        
#         def build(self, image, label):
#             logits = self.graph_model(image)
#             loss = self.criterion(logits, label)
#             loss.backward()
#             return loss
    
#     class ModelEvalGraph(flow.nn.Graph):
#         def __init__(self):
#             super().__init__()
#             self.graph_model = graph_model
        
#         def build(self, image):
#             with flow.no_grad():
#                 logits = self.graph_model(image)
#                 predictions = logits.softmax()
#             return predictions
    
#     model_train_graph = ModelTrainGraph()
#     model_eval_graph = ModelEvalGraph()
#     return model_train_graph, model_eval_graph


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
    
    def graph_train(self, train_data_loader, val_data_loader, model, criterion, optimizer):
        all_samples = len(val_data_loader) * self.args.val_batch_size
        print_interval = self.args.print_interval

        class ModelTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.graph_model = model
                self.criterion = criterion
                self.add_optimizer("sgd", optimizer)
            
            def build(self, image, label):
                logits = self.graph_model(image)
                loss = self.criterion(logits, label)
                loss.backward()
                return loss
    
        class ModelEvalGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.graph_model = model
            
            def build(self, image):
                with flow.no_grad():
                    logits = self.graph_model(image)
                    predictions = logits.softmax()
                return predictions
    
        model_train_graph = ModelTrainGraph()
        model_eval_graph = ModelEvalGraph()
        # train
        for epoch in range(self.args.epochs):
            model.train()
            start_training_time = time.time()
            for b in range(len(train_data_loader)):
                image, label = train_data_loader()

                # oneflow graph train
                iter_start_time = time.time()
                image = image.to("cuda")
                label = label.to("cuda")
                loss = model_train_graph(image, label)
                iter_end_time = time.time()
                if b % print_interval == 0:
                    l = loss.numpy()[0]
                    iter_time = iter_end_time - iter_start_time
                    print(
                        "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                            epoch, b, l, iter_time
                        )
                    )
                    self.graph_losses.append(l)
                    self.graph_train_step_time_list.append(iter_time)

            end_training_time = time.time()
            self.graph_train_epoch_time_list.append(end_training_time - start_training_time)
            print("epoch %d train done, start validation" % epoch)

            # validate
            model.eval()
            correct = 0.0
            eval_start_time = time.time()
            for b in tqdm(range(len(val_data_loader))):
                image, label = val_data_loader()
                image = image.to("cuda")
                predictions = model_eval_graph(image)
                preds = predictions.numpy()
                clsidxs = np.argmax(preds, axis=1)

                label_nd = label.numpy()
                for i in range(self.args.val_batch_size):
                    if clsidxs[i] == label_nd[i]:
                        correct += 1
            eval_end_time = time.time()
            self.graph_eval_epoch_time_list.append(eval_end_time - eval_start_time)
            top1_acc = correct / all_samples
            self.graph_acc.append(top1_acc)
            print("epoch %d, oneflow top1 val acc: %f" % (epoch, top1_acc))

    def eager_train(self, train_data_loader, val_data_loader, model, criterion, optimizer):
        all_samples = len(val_data_loader) * self.args.val_batch_size
        print_interval = self.args.print_interval

        # train
        for epoch in range(self.args.epochs):
            model.train()
            start_training_time = time.time()
            for b in range(len(train_data_loader)):
                image, label = train_data_loader()

                # oneflow graph train
                iter_start_time = time.time()
                image = image.to("cuda")
                label = label.to("cuda")
                logits = model(image)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                iter_end_time = time.time()
                if b % print_interval == 0:
                    l = loss.numpy()[0]
                    iter_time = iter_end_time - iter_start_time
                    print(
                        "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                            epoch, b, l, iter_time
                        )
                    )
                    self.eager_losses.append(l)
                    self.eager_train_step_time_list.append(iter_time)

            end_training_time = time.time()
            self.eager_train_epoch_time_list.append(end_training_time - start_training_time)
            print("epoch %d train done, start validation" % epoch)

            # validate
            model.eval()
            correct = 0.0
            eval_start_time = time.time()
            for b in tqdm(range(len(val_data_loader))):
                image, label = val_data_loader()
                image = image.to("cuda")
                with flow.no_grad():
                    logits = model(image)
                    predictions = logits.softmax()
                preds = predictions.numpy()
                clsidxs = np.argmax(preds, axis=1)

                label_nd = label.numpy()
                for i in range(self.args.val_batch_size):
                    if clsidxs[i] == label_nd[i]:
                        correct += 1
            eval_end_time = time.time()
            self.eager_eval_epoch_time_list.append(eval_end_time - eval_start_time)
            top1_acc = correct / all_samples
            self.eager_acc.append(top1_acc)
            print("epoch %d, oneflow top1 val acc: %f" % (epoch, top1_acc))    

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
    train_data_loader, val_data_loader, eager_model, graph_model, eager_optimizer, graph_optimizer, criterion = setup(args)
    print("***** Graph Training *****")
    trainer.graph_train(train_data_loader = train_data_loader, 
                        val_data_loader = val_data_loader, 
                        model = graph_model, 
                        criterion = criterion, 
                        optimizer = graph_optimizer)

    print("***** Eager Training *****")
    trainer.eager_train(train_data_loader = train_data_loader, 
                        val_data_loader = val_data_loader, 
                        model = eager_model, 
                        criterion = criterion, 
                        optimizer = eager_optimizer)
    save_result(trainer)