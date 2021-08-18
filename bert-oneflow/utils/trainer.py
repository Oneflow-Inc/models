import os

import oneflow as flow
import oneflow.nn as nn
from model.bert import BERT
from model.language_model import BERTLM

from utils.optim_schedule import ScheduledOptim


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(
        self,
        bert: BERT,
        vocab_size: int,
        train_dataloader=None,
        test_dataloader=None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=10000,
        with_cuda: bool = True,
        cuda_devices=None,
        log_freq: int = 10,
    ):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        # cuda_condition = flow.cuda.is_available() and with_cuda
        self.vocab_size = vocab_size
        cuda_condition = with_cuda
        self.device = flow.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert.to(self.device)
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)
        # self.model.load_state_dict(flow.load("output/init"))

        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and flow.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % flow.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # # Setting the Adam optimizer with hyper-param
        self.optim = flow.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optim, self.bert.hidden, n_warmup_steps=warmup_steps
        )

        self.ns_criterion = nn.NLLLoss()
        self.lm_criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, iter_per_epoch, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        # data_iter = tqdm.tqdm(
        #     range(len(iter_per_epoch)),
        #     desc="EP_%s:%d" % (str_code, epoch),
        #     total=iter_per_epoch,
        #     bar_format="{l_bar}{r_bar}",
        # )

        total_loss = 0.0
        total_correct = 0
        total_element = 0

        for i in range(len(iter_per_epoch)):
            (
                input_ids,
                next_sent_labels,
                input_masks,
                segment_ids,
                masked_lm_ids,
                masked_lm_positions,
                masked_lm_weights,
            ) = self.train_data()

            input_ids = input_ids.to(device=self.device)
            input_masks = input_masks.to(device=self.device)
            segment_ids = segment_ids.to(device=self.device)
            next_sent_labels = next_sent_labels.to(device=self.device)
            masked_lm_ids = masked_lm_ids.to(device=self.device)
            masked_lm_positions = masked_lm_positions.to(device=self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(
                input_ids, input_masks, segment_ids
            )

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            ns_loss = self.ns_criterion(next_sent_output, next_sent_labels.squeeze(1))

            # 2-2. NLLLoss of predicting masked token word
            mask_lm_output = flow.gather(
                mask_lm_output,
                index=masked_lm_positions.unsqueeze(2).repeat(1, 1, self.vocab_size),
                dim=1,
            )
            mask_lm_output = flow.reshape(mask_lm_output, [-1, self.vocab_size])

            label_id_blob = flow.reshape(masked_lm_ids, [-1])

            # 2-2. NLLLoss of predicting masked token word
            lm_loss = self.lm_criterion(mask_lm_output, label_id_blob)

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = ns_loss + lm_loss

            # 3. backward and optimization only in train
            if train:
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                self.optim_schedule.zero_grad()

            # next sentence prediction accuracy
            correct = (
                next_sent_output.argmax(dim=-1)
                .eq(next_sent_labels.squeeze(1))
                .sum()
                .numpy()
                .item()
            )
            total_loss += loss.numpy().item()
            total_correct += correct
            total_element += next_sent_labels.nelement()

            if (i + 1) % self.log_freq == 0:
                print(
                    "Epoch {}, iter {}, avg_loss {:.3f}, total_acc {:.2f}".format(
                        epoch,
                        (i + 1),
                        total_loss / (i + 1),
                        total_correct * 100.0 / total_element,
                    )
                )

        print(
            "Epoch {}, iter {}, loss {:.3f}, total_acc {:.2f}".format(
                epoch,
                (i + 1),
                total_loss / (i + 1),
                total_correct * 100.0 / total_element,
            )
        )

    def save(self, epoch, file_path="checkpoints"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = os.path.join(file_path, "epoch_%d" % epoch)
        flow.save(self.bert.state_dict(), output_path)
        print("Epoch:%d Model Saved on:" % epoch, output_path)
        return output_path
