import oneflow as flow
import argparse
import numpy as np
import os
import time
from tqdm import tqdm

from model.alexnet import alexnet
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
    parser.add_argument("--results", type=str, default="./check_results", help="tensorboard file path")
    parser.add_argument("--tag", type=str, default="training_info", help="info tag")
    parser.add_argument(
        "--print_interval", type=int, default=10, help="print info frequency"
    )
    return parser.parse_args()

def build_loader(args):
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
    return train_data_loader, val_data_loader  

def setup(args):
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

    return eager_model, graph_model, eager_optimizer, graph_optimizer, criterion

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
                    l = loss.numpy()
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
                    l = loss.numpy()
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
    print("Results saved to: ", training_results_path)

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

def save_report(trainer):
    print("***** Save Report *****")
    # folder setup
    report_path = os.path.join(args.results)
    os.makedirs(report_path, exist_ok=True)
    
    # calculate absolute loss difference
    abs_loss_diff = abs(np.array(trainer.eager_losses) - np.array(trainer.graph_losses))

    # calculate losses linear correlation
    loss_corr = calc_corr(trainer.eager_losses, trainer.graph_losses)

    # calculate accuracy linear correlation
    acc_corr = calc_corr(trainer.eager_acc, trainer.graph_acc)

    # training time compare
    train_time_compare = time_compare(trainer.graph_train_epoch_time_list, trainer.eager_train_epoch_time_list)

    # validate time compare
    val_time_compare = time_compare(trainer.graph_eval_epoch_time_list, trainer.eager_eval_epoch_time_list)

    # save report
    save_path = os.path.join(report_path, 'check_report.txt')
    writer = open(save_path, "w")
    writer.write("Check Report\n")
    writer.write("Model: Alexnet\n")
    writer.write("Check Results Between Eager Model and Graph Model\n")
    writer.write("=================================================\n")
    writer.write("Loss Correlation: %.4f\n\n" % loss_corr)
    writer.write("Max Loss Difference: %.4f\n" % abs_loss_diff.max())
    writer.write("Min Loss Difference: %.4f\n" % abs_loss_diff.min())
    writer.write("Loss Difference Range: (%.4f, %.4f)\n\n" % (abs_loss_diff.min(), abs_loss_diff.max()))
    writer.write("Accuracy Correlation: %.4f\n\n" % acc_corr)
    writer.write("Train Time Compare: %.4f (Eager) : %.4f (Graph)\n\n" % (1.0, train_time_compare))
    writer.write("Val Time Compare: %.4f (Eager) : %.4f (Graph)" % (1.0, val_time_compare))
    writer.close()
    print("Report saved to: ", save_path)


if __name__ == "__main__":
    args = _parse_args()
    trainer = Trainer(args)
    eager_model, graph_model, eager_optimizer, graph_optimizer, criterion = setup(args)
    print("***** Graph Training *****")
    print("***** Preparing Dataloader *****")
    graph_train_loader, graph_val_loader = build_loader(args)
    print("***** Done *****")
    trainer.graph_train(train_data_loader = graph_train_loader, 
                        val_data_loader = graph_val_loader, 
                        model = graph_model, 
                        criterion = criterion, 
                        optimizer = graph_optimizer)

    print("***** Eager Training *****")
    print("***** Preparing Dataloader *****")
    eager_train_loader, eager_val_loader = build_loader(args)
    print("***** Done *****")
    trainer.eager_train(train_data_loader = eager_train_loader, 
                        val_data_loader = eager_val_loader, 
                        model = eager_model, 
                        criterion = criterion, 
                        optimizer = eager_optimizer)
    
    save_result(trainer)
    save_report(trainer)