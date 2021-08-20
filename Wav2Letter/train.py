import argparse
import os
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim

from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
from Wav2Letter.decoder import GreedyDecoder


def get_args():
    parser = argparse.ArgumentParser(
        """Wav2Letter train"""
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--mfcc_features", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained_model", type=str, default="None")
    parser.add_argument("--datasets_path", type=str, default="speech_data")
    parser.add_argument("--output_path", type=str, default="save_models")

    args = parser.parse_args()
    return args


def train(opt):
    batch_size = opt.batch_size
    epochs = opt.epochs
    mfcc_features = opt.mfcc_features
    rate = opt.rate
    datasets_path = opt.datasets_path

    # load saved numpy arrays for google speech command
    gs = GoogleSpeechCommand()
    _inputs, _targets = gs.load_vectors(datasets_path)
    grapheme_count = gs.intencode.grapheme_count

    print("training google speech dataset")
    print("data size", len(_inputs))
    print("batch_size", batch_size)
    print("epochs", epochs)
    print("num_mfcc_features", mfcc_features)
    print("grapheme_count", grapheme_count)

    inputs = flow.Tensor(_inputs).to('cuda')
    targets = flow.Tensor(_targets, dtype=flow.int).to('cuda')

    # split train, eval, test
    data_size = len(_inputs)
    train_inputs = inputs[0: int(rate * data_size)]
    train_targets = targets[0: int(rate * data_size)]
    eval_inputs = inputs[int(rate * data_size): -1000]
    eval_targets = targets[int(rate * data_size): -1000]

    # Initialize model, loss, optimizer
    model = Wav2Letter(mfcc_features, grapheme_count)
    model.to('cuda')

    ctc_loss = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # load pretrained model
    if opt.pretrained_model != None:
        model.load_state_dict(flow.load(opt.pretrained_model))

    train_total_steps = int(train_inputs.size(0) // batch_size)
    eval_total_steps = int(eval_inputs.size(0) // batch_size)

    for epoch in range(epochs):
        samples_processed = 0
        avg_epoch_loss = 0

        for step in range(train_total_steps):
            train_data_batch = train_inputs[samples_processed : batch_size + samples_processed].transpose(1, 2)

            log_probs = model(train_data_batch)
            log_probs = log_probs.transpose(1, 2).transpose(0, 1)

            targets = train_targets[samples_processed: batch_size + samples_processed]

            input_lengths = flow.Tensor(np.full((batch_size,), log_probs.shape[0]), dtype=flow.int).to('cuda')
            target_lengths = flow.Tensor([target.shape[0] for target in targets], dtype=flow.int).to('cuda')

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            avg_epoch_loss += loss.numpy().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            samples_processed += batch_size

        # evaluate
        decoder = GreedyDecoder()
        wer = 0
        start_index = 0
        for step in range(eval_total_steps):
            eval_data_batch = eval_inputs[start_index : batch_size + start_index].transpose(1, 2)
            eval_targets_batch = eval_targets[start_index : batch_size + start_index]
            eval_log_props = model(eval_data_batch)

            output = decoder.decode(eval_log_props)
            pred_strings, output = decoder.convert_to_strings(output)
            eval_target_strings = decoder.convert_to_strings(eval_targets_batch, remove_repetitions=False, return_offsets=False)
            wer += decoder.wer(eval_target_strings, pred_strings)
            start_index += batch_size
        
        print("epoch", epoch + 1, "average epoch loss", avg_epoch_loss / train_total_steps, "wer", wer/eval_total_steps)

        # save models
        if (epoch + 1) % 100 == 0:
            flow.save(model.state_dict(), os.path.join(opt.output_path, "model_{}.pth".format(epoch+1)))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
