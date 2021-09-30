import os
import shutil
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.data import Dataset, DataLoader
import extract_feats.opensmile as of
import extract_feats.librosa as lf
from utils import parse_opt, curve
from models import lstm_ser, cnn1d_ser


class SpeechDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None, target_transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.length = y_data.shape[0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x_data[index]
        label = self.y_data[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label


def train_eval(config):
    """
    training and testing model
    Args:
        config: configuration items
    Returns:
        the trained model and the evaluation results
    """
    # loading the features preprocessed by preprocess.py
    if config.feature_method == "o":
        x_train, x_test, y_train, y_test = of.load_feature(
            config, config.train_feature_path_opensmile, train=True
        )

    elif config.feature_method == "l":
        x_train, x_test, y_train, y_test = lf.load_feature(
            config, config.train_feature_path_librosa, train=True
        )

    n_feats = x_train.shape[1]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    train_dataset = SpeechDataset(x_train, y_train)
    test_dataset = SpeechDataset(x_test, y_test)
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    if config.model == "lstm":
        model = lstm_ser(
            n_feats, config.rnn_size, len(config.class_labels), config.batch_size
        )
    else:
        model = cnn1d_ser(
            1, config.n_kernels, n_feats, config.hidden_size, len(config.class_labels)
        )

    loss_fn = nn.CrossEntropyLoss()
    model.to("cuda")
    loss_fn.to("cuda")
    optimizer = flow.optim.Adam(model.parameters(), lr=config.lr)

    def train(iter, model, loss_fn, optimizer):
        size = len(iter.dataset)
        num_batches = len(iter)
        trian_loss, correct = 0, 0
        for batch, (x, y) in enumerate(iter):
            x = x.reshape(1, x.shape[0], x.shape[1])
            x = flow.tensor(x, dtype=flow.float32, device="cuda")
            y = flow.tensor(y, dtype=flow.int32, device="cuda")
            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)
            bool_value = np.argmax(pred.numpy(), 1) == y.numpy()
            correct += float(bool_value.sum())
            trian_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current = batch * config.batch_size
            if batch % 15 == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return trian_loss / num_batches, 100 * correct / size

    def test(iter, model, loss_fn):
        size = len(iter.dataset)
        num_batches = len(iter)
        model.eval()
        test_loss, correct = 0, 0
        flag = 0
        with flow.no_grad():
            for x, y in iter:
                if x.shape[0] != config.batch_size:
                    flag = 1
                    n = config.batch_size - x.shape[0]
                    x_comp = flow.zeros((n, x.shape[1]))
                    y_comp = flow.zeros(y.shape[0])
                    x = flow.tensor(np.vstack((x.numpy(), x_comp.numpy())))
                    y = flow.tensor(np.hstack((y.numpy(), y_comp.numpy())))

                x = x.reshape(1, x.shape[0], x.shape[1])
                x = flow.tensor(x, dtype=flow.float32, device="cuda")
                y = flow.tensor(y, dtype=flow.int32, device="cuda")

                pred = model(x)

                test_loss += loss_fn(pred, y)
                if flag == 0:
                    bool_value = np.argmax(pred.numpy(), 1) == y.numpy()
                else:
                    bool_value = np.argmax(pred.numpy()[0:16], 1) == y.numpy()[0:16]

                correct += float(bool_value.sum())
        test_loss /= num_batches
        print("test_loss", test_loss, "num_batches ", num_batches)
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}"
        )

        return test_loss, 100 * correct

    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for e in range(config.epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        tr_loss, tr_acc = train(train_iter, model, loss_fn, optimizer)
        train_loss.append(tr_loss.numpy())
        train_acc.append(tr_acc)
        te_loss, te_acc = test(test_iter, model, loss_fn)
        test_loss.append(te_loss.numpy())
        test_acc.append(te_acc)
    print("Done!")

    # Saving the trained model
    model_path = os.path.join(config.checkpoint_path, config.checkpoint_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    flow.save(model.state_dict(), model_path)

    # Visualize the training process
    if config.vis:
        curve(train_acc, test_acc, "Accuracy", "acc")
        curve(train_loss, test_loss, "Loss", "loss")

    return train_loss, test_loss, train_acc, test_acc


config = parse_opt()
train_eval(config)
