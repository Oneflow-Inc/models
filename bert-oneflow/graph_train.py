import argparse
import tqdm 


import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.data import DataLoader

from model.language_model import BERTLM
from model.bert import BERT
from trainer.pretrain import BERTTrainer
from dataset.dataset import BERTDataset
from dataset.vocab import WordVocab


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--train_dataset",
        required=False,
        type=str,
        default="data/corpus.small",
        help="train dataset for train bert",
    )
    parser.add_argument(
        "-t",
        "--test_dataset",
        type=str,
        default="data/corpus.small",
        help="test set for evaluate train set",
    )
    parser.add_argument(
        "-v",
        "--vocab_path",
        required=False,
        default="data/vocab.small",
        type=str,
        help="built vocab model path with bert-vocab",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        default="output/bert.model",
        type=str,
        help="ex)output/bert.model",
    )

    parser.add_argument(
        "-hs",
        "--hidden",
        type=int,
        default=256,
        help="hidden size of transformer model",
    )
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument(
        "-a", "--attn_heads", type=int, default=8, help="number of attention heads"
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, default=20, help="maximum sequence len"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="number of batch_size"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=0, help="dataloader worker size"
    )

    parser.add_argument(
        "--with_cuda",
        type=bool,
        default=True,
        help="training with CUDA: true, or false",
    )
    parser.add_argument(
        "--corpus_lines", type=int, default=None, help="total number of lines in corpus"
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )
    parser.add_argument(
        "--on_memory", type=bool, default=True, help="Loading on memory: true or false"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam first beta value"
    )

    args = parser.parse_args()

    if(args.with_cuda): 
        device = flow.device("cuda:0")
    else: 
        device = flow.device("cpu")
    
    print("Device is: ", device)


    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(
        args.train_dataset,
        vocab,
        seq_len=args.seq_len,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory,
    )

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = (
        BERTDataset(
            args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory
        )
        if args.test_dataset is not None
        else None
    )

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_data_loader = (
        DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if test_dataset is not None
        else None
    )

    print("Building BERT model")
    bert_module = BERT(
        len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    bert_module.to(device)
    
    bert_model = BERTLM(bert_module, len(vocab))
    bert_model.to(device)

    # TODO: Adam optimizer 'generate_conf_for_graph' function
    # of_adam = flow.optim.Adam(bert_model.parameters(), 
    #                           lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), 
    #                           weight_decay=args.adam_weight_decay)
    of_adam = flow.optim.SGD(bert_model.parameters(), 
                             lr=args.lr, 
                             weight_decay=args.adam_weight_decay)



    # TODO：how to add this schedule in
    # optim_schedule = ScheduledOptim(
    #         self.optim, self.bert.hidden, n_warmup_steps=warmup_steps
    #     )

    # of_nll_loss = nn.NLLLoss(ignore_index=0)
    of_nll_loss = nn.NLLLoss()

    of_nll_loss.to(device)

    class BertGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self.nll_loss = of_nll_loss
            self.add_optimizer("adam", of_adam)
        
        def build(self, bert_input, segment_label, is_next, bert_label):
            next_sent_output, mask_lm_output = self.bert(bert_input, segment_label)
            
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.nll_loss(next_sent_output, is_next)
            
            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.nll_loss(
                mask_lm_output.transpose(1, 2), bert_label
            )
            
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            loss.backward()
            return next_sent_output, mask_lm_output, loss

    bert_graph = BertGraph()

    # print("Creating BERT Trainer")
    # trainer = BERTTrainer(
    #     bert,
    #     len(vocab),
    #     train_dataloader=train_data_loader,
    #     test_dataloader=test_data_loader,
    #     lr=args.lr,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     with_cuda=args.with_cuda,
    #     cuda_devices=args.cuda_devices,
    #     log_freq=10,
    # )

    print_interval = 10

    for epoch in range(args.epochs):
        bert_model.train()

        data_iter = tqdm.tqdm(
            enumerate(train_data_loader),
            desc="EP_%s:%d" % ("train", epoch),
            total=len(train_data_loader),
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            for key, value in data.items():
                if key == "is_next":
                    # print("value shape is: ", value.shape)
                    # value = value.squeeze(1)
                    value = value.squeeze(0)

                data[str(key)] = flow.Tensor(
                    value.numpy(), dtype=flow.int64, device=device
                )
            # print("Device is: ", device)
            #     # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(device=device) for key, value in data.items()}

            print("Graph Before ==== ")
            next_sent_output, mask_lm_output, loss = bert_graph(data["bert_input"], data["segment_label"], data["is_next"], data["bert_label"])
            print("Graph After !!!! ")

            # flow.save(self.bert.state_dict(), "checkpoints/bert_%d_loss_%f" % (i, loss.numpy().item()))

            # next sentence prediction accuracy
            correct = (
                next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().numpy().item()
            )
            avg_loss += loss.numpy().item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            # post_fix = {
            #     "epoch": epoch,
            #     "iter": i,
            #     "avg_loss": avg_loss / (i + 1),
            #     "avg_acc": total_correct / total_element * 100,
            #     "loss": loss.numpy().item(),
            # }

            # if i % self.log_freq == 0:
            #     data_iter.write(str(post_fix))

        print("total_correct >>>>>>>>>>>>>> ", total_correct)
        print("total_element >>>>>>>>>>>>>> ", total_element)
        print(
            "EP%d_%s, avg_loss=" % (epoch, str_code),
            avg_loss / len(data_iter),
            "total_acc=",
            total_correct * 100.0 / total_element,
        )


        # trainer.train(epoch)
        # print("Saving model...")
        # trainer.save(epoch, args.output_path)
        # if test_data_loader is not None:
        #     print("Running testing...")
        #     trainer.test(epoch)


main()
