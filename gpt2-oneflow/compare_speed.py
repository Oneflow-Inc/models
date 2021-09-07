import argparse

import time
import oneflow as flow
from oneflow.utils.data import DataLoader

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel as t_GPT2LMHeadModel, GPT2Config as t_GPT2Config

from model_config import GPT2Config
from model import GPT2LMHeadModel
from pt_model import GPT2LMHeadModel as pt_GPT2LMHeadModel
from trainer import Trainer
from gpt_dataset import GPTDataset
from tokenizer import build_tokenizer

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", required=False, type=str, default="data/corpus.small", help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="data/corpus.small", help="test set for evaluation")
    parser.add_argument("--vocab_file", required=False, default="vocab.json", type=str)
    parser.add_argument("--merges_file", required=False, default="merge.txt", type=str)
    parser.add_argument("--output_path", required=False, default="output/model", type=str, help="save path")

    parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument("--batch_size", type=int, default=2, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    # parser.add_argument("--device_id", type=int, default=2)

    args = parser.parse_args()

    # device = flow.device(f"cuda:{args.device_id}" if args.device_id >= 0 else "cpu")

    print("building tokenizer")
    tokenizer = build_tokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file, tokenizer_type="GPT2BPETokenizer")

    print("building train dataset")
    train_dataset = GPTDataset(args.train_dataset, tokenizer, args.seq_len)
    
    print("building train dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for i, b in enumerate(train_data_loader):
        if i == 2:
            batch = b
            break

    of_batch = flow.Tensor(batch, dtype=flow.long).cuda()

    print("building model")
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    # from convert_pt_ckpt_to_of import convert_pt_checkpoint_to_of
    # convert_pt_checkpoint_to_of(model, "test_gpt2_model.pt", "test_gpt2_oneflow_model")

    model.load_state_dict(flow.load("test_gpt2_oneflow_model"))
    # model.lm_head.weight = model.transformer.wte.weight
    
    model.cuda()
    model.eval()

    # of_parameters_value = [param.numpy() for param in list(model.parameters())]
    optimizer = flow.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = flow.optim.Adam(model.parameters(), lr=0.001, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay)

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0
    of_loss = list()

    print("start oneflow training loop....")
    start_t = time.time()
    for epoch in range(args.epochs):
        s_t = time.time()
        loss = model(of_batch, labels=of_batch)[0]
        # loss = model(of_batch, None, None, of_batch)[0]
        for_time += time.time() - s_t
        
        of_loss.append(loss.numpy())

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        of_grad = model.transformer.h[0].attn.c_attn.weight.grad

        s_t = time.time()
        optimizer.step()
        optimizer.zero_grad()
        update_time += time.time() - s_t

    # of_loss = loss.numpy()
    end_t = time.time()

    print("oneflow traning loop avg time : {}".format((end_t - start_t) / args.epochs))
    print("forward avg time : {}".format(for_time / args.epochs))
    print("backward avg time : {}".format(bp_time / args.epochs))
    print("update parameters avg time : {}".format(update_time / args.epochs))

    of_parameters_names = []
    of_parameters_value = []
    for name, param in model.named_parameters():
        of_parameters_names.append(name)
        of_parameters_value.append(param.numpy())

    torch.cuda.empty_cache()

    # pt_device = torch.device(f"cuda:{args.device_id}" if args.device_id >= 0 else "cpu")

    pt_batch = torch.from_numpy(batch.numpy()).long().cuda()

    pt_model = pt_GPT2LMHeadModel(config)
    pt_model.load_state_dict(torch.load("test_gpt2_model.pt"))
    # pt_model.lm_head.weight = pt_model.transformer.wte.weight

    # torch.save(pt_model.state_dict(), f="test_gpt2_model.pt")

    # pt_model = t_GPT2LMHeadModel.from_pretrained('gpt2')
    # pt_embs = pt_model.transformer.wte.weight
    # pt_lin = torch.nn.Linear(pt_embs.size()[1], pt_embs.size()[0], bias=False)
    # pt_lin.weight = pt_embs
    # pt_model.set_output_embeddings(pt_lin)
    # print(pt_embs.size())
    # pt_model.lm_head.weight = pt_model.transformer.wte.weight


    pt_model.cuda()
    pt_model.eval()
    
    learning_rate = 0.01
    mom = 0.9
    pt_optimizer = torch.optim.SGD(pt_model.parameters(), lr=0.001)
    # pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay)

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0
    pt_loss = list()
    loss = None
    print("start pytorch training loop....")
    start_t = time.time()
    for epoch in range(args.epochs):
        s_t = time.time()
        loss = pt_model(pt_batch, labels=pt_batch)[0]
        for_time += time.time() - s_t
        
        pt_loss.append(loss.cpu().detach().numpy())

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        pt_grad = pt_model.transformer.h[0].attn.c_attn.weight.grad

        s_t = time.time()
        pt_optimizer.step()
        pt_optimizer.zero_grad()
        update_time += time.time() - s_t

    end_t = time.time()

    print("pytorch traning loop avg time : {}".format((end_t - start_t) / args.epochs))
    print("forward avg time : {}".format(for_time / args.epochs))
    print("backward avg time : {}".format(bp_time / args.epochs))
    print("update parameters avg time : {}".format(update_time / args.epochs))

    pt_parameters_names = []
    pt_parameters_value = []
    for name, param in pt_model.named_parameters():
        pt_parameters_names.append(name)
        pt_parameters_value.append(param.cpu().detach().numpy())

    # print((of_grad.numpy() == pt_grad.cpu().detach().numpy()).all())

    torch.cuda.empty_cache()

    assert len(of_parameters_names) == len(pt_parameters_names)
    for i in range(len(of_parameters_names)):
        if of_parameters_names[i] == pt_parameters_names[i]:
            if (of_parameters_value[i] == pt_parameters_value[i]).all():
                print(of_parameters_names[i], "is same!")
            else:
                print(of_parameters_names[i], "is NOT same!")
        else:
            print(of_parameters_names[i], "cannot found!")

    # print(set(of_parameters_value).symmetric_difference(set(pt_parameters_value)))

    # t_config = t_GPT2Config()
    # t_model = t_GPT2LMHeadModel(t_config)
    # t_model.load_state_dict(torch.load("test_gpt2_model.pt"))
    # t_model.lm_head.weight = t_model.transformer.wte.weight

    # t_model.to(pt_device)
    # t_model.eval()

    # learning_rate = 0.01
    # mom = 0.9
    # # pt_optimizer = torch.optim.SGD(pt_model.parameters(), lr=1e-4, momentum=0.9)
    # pt_optimizer = torch.optim.Adam(t_model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay)

    # for_time = 0.0
    # bp_time = 0.0
    # update_time = 0.0
    # tr_loss = list()
    # loss = None
    # print("start transformers training loop....")
    # start_t = time.time()
    # for epoch in range(args.epochs):
    #     s_t = time.time()
    #     loss = t_model(pt_batch, labels=pt_batch)[0]
    #     for_time += time.time() - s_t
        
    #     tr_loss.append(loss.cpu().detach().numpy())

    #     s_t = time.time()
    #     loss.backward()
    #     bp_time += time.time() - s_t

    #     s_t = time.time()
    #     pt_optimizer.step()
    #     pt_optimizer.zero_grad()
    #     update_time += time.time() - s_t

    # end_t = time.time()

    # print("transformers pytorch traning loop avg time : {}".format((end_t - start_t) / args.epochs))
    # print("forward avg time : {}".format(for_time / args.epochs))
    # print("backward avg time : {}".format(bp_time / args.epochs))
    # print("update parameters avg time : {}".format(update_time / args.epochs))



    for i in range(args.epochs):
        print(i, of_loss[i], pt_loss[i])

    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    epochs = np.arange(1, args.epochs + 1)

    plt.plot(epochs, of_loss, label="oneflow")
    plt.plot(epochs, pt_loss, label="pytorch")
    plt.legend()
    plt.savefig("./1.jpg")
    # plt.show()

if __name__=='__main__':
    main()
