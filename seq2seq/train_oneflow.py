from models.seq_seq_oneflow import *
from utils.utils_oneflow import *
import oneflow.optim as optim
import oneflow.nn as nn
import time
import random
import argparse


# refer to: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def _parse_args():
    parser = argparse.ArgumentParser("flags for train seq2seq")

    parser.add_argument(
        "--device", type=str, default="cuda", help="device",
    )

    parser.add_argument(
        "--save_encoder_checkpoint_path",
        type=str,
        default="./saving_model_oneflow/encoder/",
        help="save checkpoint encoder dir",
    )

    parser.add_argument(
        "--save_decoder_checkpoint_path",
        type=str,
        default="./saving_model_oneflow/decoder/",
        help="save checkpoint decoder dir",
    )

    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")

    parser.add_argument("--n_iters", type=int, default=75000, help="num of iters")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")

    parser.add_argument(
        "--drop", type=float, default=0.1, help="the dropout of decoder_embedding"
    )

    return parser.parse_args()


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=MAX_LENGTH,
):
    encoder_hidden = encoder.init_Hidden().to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = []

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs.append(encoder_output[0])

    if len(encoder_outputs) != MAX_LENGTH:
        for _ in range(MAX_LENGTH - len(encoder_outputs)):
            encoder_outputs.append(flow.zeros((1, 256)).to(device))

    encoder_outputs = flow.cat(encoder_outputs, dim=0)
    decoder_input = flow.tensor([[SOS_token]]).to(device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.numpy() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.numpy() / target_length


def trainIters(
    encoder,
    decoder,
    n_iters,
    pairs,
    input_lang,
    output_lang,
    print_every=1000,
    plot_every=100,
    learning_rate=0.01,
):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensorsFromPair(random.choice(pairs), input_lang, output_lang)
        for _ in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, iter / n_iters),
                    iter,
                    iter / n_iters * 100,
                    print_loss_avg,
                )
            )

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def main(args):

    device = args.device

    # pre
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)
    # training
    hidden_size = args.hidden_size
    encoder = EncoderRNN_oneflow(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN_oneflow(
        hidden_size, output_lang.n_words, dropout_p=args.drop
    ).to(device)
    trainIters(
        encoder,
        attn_decoder,
        args.n_iters,
        pairs,
        input_lang,
        output_lang,
        print_every=5000,
        plot_every=100,
        learning_rate=args.lr,
    )
    # saving model...'
    flow.save(encoder.state_dict(), args.save_encoder_checkpoint_path)
    flow.save(attn_decoder.state_dict(), args.save_decoder_checkpoint_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
