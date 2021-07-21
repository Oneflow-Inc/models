import oneflow.experimental as flow

import numpy as np
from tqdm import tqdm 
import shutil
import os

def batch_loader(data, 
                label, 
                batch_size,
                shuffle = True):
    if shuffle:
        permu = np.random.permutation(len(data))
        data, label = data[permu], label[permu]
    batch_n = len(data) // batch_size
    x_batch = [flow.tensor(data[i * batch_size:(i * batch_size + batch_size)],dtype = flow.long) for i in range(batch_n)]
    y_batch = [flow.tensor(label[i * batch_size:(i * batch_size + batch_size)],dtype = flow.long)for i in range(batch_n)]
    if batch_size*batch_n < len(data):
        x_batch += [flow.tensor(data[batch_size*batch_n:len(label)],dtype = flow.long)]
        y_batch += [flow.tensor(label[batch_size*batch_n:len(label)],dtype = flow.long)]
    return x_batch, y_batch

def train(model,
          device,
          train_data,
          dev_data,
          loss_func,
          optimizer,
          epochs,
          train_batch_size,
          eval_batch_size,
          save_path):
    global_acc = float('-inf')
    for i in range(epochs):
        x_batch, y_batch = batch_loader(train_data[0],train_data[1],train_batch_size)
        model.train()
        model.training=True
        training_loss = 0
        all_res,all_ground_truths = [],[]
        total_correct = 0
        total_wrongs = 0
        for idx,(data,label) in enumerate(tqdm(zip(x_batch, y_batch),total = len(x_batch))):
            data = data.to(device)
            label = label.to(device)
            logits = model(data)
            res = flow.argmax(logits,dim=1)
            total_correct += (res.numpy() == label.numpy()).sum()
            all_res.append(res)
            all_ground_truths.append(label)
            label = flow.tensor(np.eye(2)[label.numpy()],dtype=flow.float32).to(device)
            loss = loss_func(logits,label)
            training_loss += loss.numpy()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        all_ground_truths = flow.cat(all_ground_truths)
        train_acc = total_correct / len(all_ground_truths.numpy())
        acc = _eval(model,
                   dev_data,   
                   device,
                   eval_batch_size
                )
        if acc > global_acc:
            global_acc = acc
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            flow.save(model.state_dict(), save_path)
        print(f'[Epoch{i}] training loss: {training_loss/(idx+1)}  training accuracy: {train_acc} evaluation accuracy: {acc}')
    
def _eval(model,
          dev_data,
          device,
          batch_size = 32):
    model.eval()
    model.training=False
    x_batch, y_batch = batch_loader(dev_data[0], 
                                    dev_data[1], 
                                    batch_size,
                                    shuffle = False)
    all_res,all_ground_truths = [],[]
    total_correct = 0
    for data, label in tqdm(zip(x_batch,y_batch),total = len(x_batch)):
        with flow.no_grad():
            data = data.to(device)
            label = label.to(device)
            logits = model(data)
            res = flow.argmax(logits,dim=1)
            total_correct += (res.numpy() == label.numpy()).sum()
            all_res.append(res)
            all_ground_truths.append(label)
    all_res = flow.cat(all_res)
    all_ground_truths = flow.cat(all_ground_truths)
    acc = total_correct / len(all_ground_truths.numpy())
    return acc