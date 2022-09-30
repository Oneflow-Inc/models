import argparse
import datetime

import oneflow as of
import oneflow.nn as nn

from DataSet import MyDataset2, MyDataset1


class CELoss_ls(nn.Module):

    def __init__(self, label_smoothing):
        super(CELoss_ls, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, y):
        pred = nn.LogSoftmax(dim=1)(pred)
        temp1 = -of.gather(pred, dim=1, index=y.view(-1, 1))
        temp2 = -of.mean(pred, dim=1)
        loss = (1 - self.label_smoothing) * temp1.view(-1) + self.label_smoothing * temp2
        return of.mean(loss, dim=0)


def _parse_args():

    parser = argparse.ArgumentParser("flags for train model")

    parser.add_argument(
        "--model_layer",
        type=int,
        default=161,
        help="model layer",
    )

    parser.add_argument(
        "--dataset_method",
        type=int,
        default=1,
        help='data set loading method'
    )

    parser.add_argument(
        "--image_train_json",
        type=str,
        default="train_list.json",
        help="input image train json file")

    parser.add_argument(
        "--image_val_json",
        type=str,
        default="val_list.json",
        help="input image val json file")

    parser.add_argument(
        "--label_soft",
        type=bool,
        default=True,
        help="use label soft")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="model device",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=20,
        help="model train batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="model train lr",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="model train epochs",
    )

    parser.add_argument(
        "--standard_acc",
        type=float,
        default=0.97,
        help="save model standard acc",
    )

    return parser.parse_args()

def main(args):

    model_layer = args.model_layer

    if model_layer == 121:
        from DenseNet import DenseNet121_pre
        model = DenseNet121_pre()
    elif model_layer == 169:
        from DenseNet import DenseNet169_pre
        model = DenseNet169_pre()
    elif model_layer == 201:
        from DenseNet import DenseNet201_pre
        model = DenseNet201_pre()
    else:
        from DenseNet import DenseNet161_pre
        model = DenseNet161_pre()

    dataset_method = args.dataset_method
    image_train_json = args.image_train_json
    image_val_json = args.image_val_json

    if dataset_method == 1:
        train_dataset = MyDataset1(image_train_json)
        val_dataset = MyDataset1(image_val_json, if_train=False)
    else:
        train_dataset = MyDataset2(image_train_json)
        val_dataset = MyDataset2(image_val_json, if_train=False)
    if args.label_soft:
        lossf = CELoss_ls(label_smoothing=0.1) # using label soft
    else:
        lossf = nn.CrossEntropyLoss()

    day = datetime.date.today().day
    month = datetime.date.today().month
    val_len = val_dataset.__len__()

    device = args.device

    model.to(device)

    batch_size = args.train_batch_size
    train = of.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val = of.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=20)

    lr = args.lr
    optim = of.optim.Adam(model.parameters(), lr=lr)
    lossf = lossf.to(device)

    epochs = args.epochs
    standard_acc = args.standard_acc
    for epoch in range(1, epochs + 1):
        model.train()
        print('epoch:', epoch)
        for data, label in train:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = lossf(pred, label)
            print('loss:', loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with of.no_grad():
            sum1 = 0
            for data, label in val:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                _, top1 = of.topk(pred.softmax(dim=1), dim=1, k=1)
                sum1 = sum1 + (top1.eq(label.view(-1, 1))).sum().item()
            acc1 = sum1 / val_len
            print('the acc of val is {}'.format(acc1))
            if acc1 > standard_acc:
                of.save(model.state_dict(), './{:02d}{:02d}/161_pre_{:03d}_{:0.4f}'.format(month, day, epoch, acc1))


if __name__ == '__main__':
    args = _parse_args()
    main(args)
