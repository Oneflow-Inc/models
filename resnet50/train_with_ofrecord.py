import oneflow.experimental as flow

import numpy as np
import time
import argparse
import os
import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.ofrecord_data_utils import OFRecordDataLoader

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--save_checkpoint_path", type=str, default="./checkpoints", help="checkpoint root dir"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./ofrecord", help="dataset path"
    )
    return parser.parse_args()

def main(args):
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    #############################################
    train_batch_size = 16
    val_batch_size = 16

    train_data_loader = OFRecordDataLoader(
                            ofrecord_root = "./ofrecord",
                            mode = "train",
                            dataset_size = 9469,
                            batch_size = train_batch_size)

    val_data_loader = OFRecordDataLoader(
                            ofrecord_root = "./ofrecord",
                            mode = "val",
                            dataset_size = 3925,
                            batch_size = val_batch_size)

    epochs = 1000
    learning_rate = 0.001
    mom = 0.9

    ###############################
    # pytorch init
    torch_res50_module = pytorch_resnet50.resnet50()
    start_t = time.time()
    print(type(start_t))
    torch_params = torch_res50_module.state_dict()
    end_t = time.time()
    print('torch load params time : {}'.format(end_t - start_t))
    torch_res50_module.to('cuda')
    torch_sgd = torch.optim.SGD(torch_res50_module.parameters(), lr=learning_rate, momentum=mom)
    
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to('cuda')
    ###############################

    #################
    # oneflow init
    start_t = time.time()
    res50_module = resnet50()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    start_t = time.time()
    torch_keys = torch_params.keys()

    dic = res50_module.state_dict()
    for k in dic.keys():
        if k in torch_keys:
            dic[k] = torch_params[k].numpy()
    res50_module.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    of_corss_entropy = flow.nn.CrossEntropyLoss()

    res50_module.to('cuda')
    of_corss_entropy.to('cuda')

    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)


    ############################
    of_losses = []
    torch_losses = []

    all_samples = len(val_data_loader) * val_batch_size

    for epoch in range(epochs):
        res50_module.train()
        torch_res50_module.train()

        for b in range(len(train_data_loader)):
        # for b in range(100):
            print("epoch %d train iter %d" % (epoch, b))
            image, label = train_data_loader.get_batch()
        
            # oneflow train 
            start_t = time.time()
            image = image.to('cuda')
            label = label.to('cuda')
            logits = res50_module(image)
            loss = of_corss_entropy(logits, label)
            loss.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            end_t = time.time()
            l = loss.numpy()[0]
            of_losses.append(l)
            print('oneflow loss {}, train time : {}'.format(l, end_t - start_t))

            # pytroch train
            start_t = time.time()
            image = torch.tensor(image.numpy()).to('cuda')
            label = torch.tensor(label.numpy(), dtype=torch.long, requires_grad=False).to('cuda')
            logits = torch_res50_module(image)
            loss = corss_entropy(logits, label)
            loss.backward()
            torch_sgd.step()
            torch_sgd.zero_grad()
            end_t = time.time()
            l = loss.cpu().detach().numpy()
            torch_losses.append(l)
            print('pytorch loss {}, train time : {}'.format(l, end_t - start_t))

        print("epoch %d train done, start validation" % epoch)

        res50_module.eval()
        torch_res50_module.eval()
        correct_of = 0.0
        correct_torch = 0.0
        for b in range(len(val_data_loader)):
        # for b in range(100):
            print("epoch %d val iter %d" % (epoch, b))
            image, label = val_data_loader.get_batch()

            start_t = time.time()
            image = image.to('cuda')
            with flow.no_grad():
                logits = res50_module(image)
                predictions = logits.softmax()
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()
            for i in range(val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()
            print("of predict time: %f, %d" % (end_t - start_t, correct_of))

            # pytroch val
            start_t = time.time()
            image = torch.from_numpy(image.numpy()).to('cuda')
            with torch.no_grad():
                logits = torch_res50_module(image)
                predictions = logits.softmax(-1)
            torch_predictions = predictions.cpu().detach().numpy()
            clsidxs = np.argmax(torch_predictions, axis=1)
            for i in range(val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_torch += 1
            end_t = time.time()
            print("torch predict time: %f, %d" % (end_t - start_t, correct_torch))

        print("epoch %d, oneflow top1 val acc: %f, torch top1 val acc: %f" % (epoch, correct_of / all_samples, correct_torch / all_samples))
        
        # flow.save(res50_module.state_dict(), os.path.join(args.save_checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, correct_of / all_samples)))

    writer = open("of_losses.txt", "w")
    for o in of_losses:
        writer.write("%f\n" % o)
    writer.close()

    writer = open("torch_losses.txt", "w")
    for o in torch_losses:
        writer.write("%f\n" % o)
    writer.close()



if __name__ == "__main__":
    args = _parse_args()
    main(args)








