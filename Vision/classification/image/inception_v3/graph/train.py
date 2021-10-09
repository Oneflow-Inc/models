import os
import time
import numpy as np
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as DDP

from graph.config import get_args
from graph.data import make_data_loader
from graph.build_graph import build_train_graph, build_eval_graph
from models.inceptionv3 import inception_v3

def tensor_to_local(tensor):
    """ to local """
    tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local()
    return tensor

def tensor_to_numpy(tensor):
    """ tensor to numpy """
    tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
    return tensor

class Accuracy(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        top1_num = flow.zeros(1, dtype=flow.float32)
        num_samples = 0
        for pred, label in zip(preds, labels):
            clsidxs = pred.argmax(dim=-1)
            clsidxs = clsidxs.to(flow.int32)
            match = (clsidxs == label).sum()
            top1_num += match.to(device=top1_num.device, dtype=top1_num.dtype)
            num_samples += np.prod(label.shape).item()

        top1_acc = top1_num / num_samples
        return top1_acc

def calc_acc(preds, labels):
    correct_of = 0.0
    num_samples = 0
    for pred, label in zip(preds, labels):
        clsidxs = np.argmax(pred, axis=1)
        correct_of += (clsidxs == label).sum()
        num_samples += label.size

    top1_acc = correct_of / num_samples
    return top1_acc
   
class Trainer(object):
    def __init__(self):
        args = get_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()

        self.current_epoch = 0
        self.current_step = 0
        self.current_batch = 0
        self.is_consistent = True
        self.is_train = False

        self.model = inception_v3()
        self.init_model()
        self.criterion = flow.nn.CrossEntropyLoss(reduction="mean")
        self.train_data_loader = make_data_loader(
            args, "train", self.is_consistent, self.synthetic_data
        )
        self.val_data_loader = make_data_loader(
            args, "val", self.is_consistent, self.synthetic_data
        )
        self.optimizer = flow.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        self.acc = Accuracy()

        self.train_graph = build_train_graph(self.model, self.criterion, self.train_data_loader, self.optimizer)
        self.eval_graph = build_eval_graph(self.model, self.val_data_loader)
    
    def init_model(self):
        if self.rank in [-1, 0]:
            print("***** Model Init *****")
        start_t = time.perf_counter()

        placement = flow.env.all_device_placement("cuda")
        self.model = self.model.to_consistent(
            placement=placement, sbp=flow.sbp.broadcast
        )
        self.load_state_dict()
        end_t = time.perf_counter()
        if self.rank in [-1, 0]:
            print(f"***** Model Init Finish, time escapled: {end_t - start_t:.5f} s *****")

    def load_state_dict(self):
        # self.logger.print(f"Loading model from {self.load_path}", print_ranks=[0])
        if self.rank in [-1, 0]:
            print(f"Loading model from {self.load_path}")
        if self.load_path:
            state_dict = flow.load(self.load_path, consistent_src_rank=0)
            self.model.load_state_dict(state_dict)
    
    def train(self):
        for _ in range(self.num_epochs):
            self.train_one_epoch()
            if self.current_batch == self.total_batches:
                break
            acc = self.eval()
            save_dir = f"epoch_{self.cur_epoch}_val_acc_{acc}"
            self.save(save_dir)
            self.cur_epoch += 1
            self.cur_iter = 0
    
    def train_one_epoch(self):
        self.model.train()
        self.is_train = True

        for _ in range(self.batches_per_epoch):
            loss, pred, label = self.train_graph()
        
            self.current_step += 1
            loss = tensor_to_local(loss)
            print(loss)
            self.current_batch += 1
            
            if self.current_batch == self.total_batches:
                break
    
    def eval(self):
        self.model.eval()
        self.is_train = False
        preds, labels = [], []
        for _ in range(self.val_batches_per_epoch):
            pred, label = self.eval_graph()
            preds.append(tensor_to_numpy(pred))
            labels.append(tensor_to_numpy(label))
        top1_acc = calc_acc(preds, labels)
        print(top1_acc)
        return top1_acc

    def inference(self):
        image, label = self.val_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        with flow.no_grad():
            logits = self.model(image)
            pred = logits.softmax()

        return pred, label

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()