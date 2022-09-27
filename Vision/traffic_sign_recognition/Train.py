from DataSet import MyDataset2
from DenseNet import DenseNet161_pre
import oneflow as of
import oneflow.nn as nn
import datetime

# label soft
class CELoss_ls(nn.Module):

    def __init__(self, label_smoothing):
        super(CELoss_ls, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, y):
        pred = nn.LogSoftmax(dim=1)(pred)
        temp1 = -of.gather(pred, dim=1, index=y)
        temp2 = -of.mean(pred, dim=1)
        loss = (1 - self.label_smoothing) * temp1.view(-1) + self.label_smoothing * temp2
        return of.mean(loss, dim=0)


def train(model, train_dataset, val_dataset, lossf, lr=3e-4, epochs=50, standard_acc=0.97, device='cuda'):
    from oneflow.utils import data

    day = datetime.date.today().day
    month = datetime.date.today().month

    model.to(device)

    val_len = val_dataset.__len__()

    train = data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    val = data.DataLoader(val_dataset, shuffle=False, batch_size=20)

    optim = of.optim.Adam(model.parameters(), lr=lr)
    lossf = lossf.to(device)

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
    model = DenseNet161_pre(num_classes=10)
    train_dataset = MyDataset2(json_path='train_list.json')
    val_dataset = MyDataset2(json_path='val_list.json', if_train=False)
    lossf = nn.CrossEntropyLoss()
    # lossf = (CELoss_ls(label_smoothing=0.1))  # using label soft
    train(model, train_dataset, val_dataset, lossf)
