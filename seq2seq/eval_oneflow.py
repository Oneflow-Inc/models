from models.seq_seq_oneflow import AttnDecoderRNN_oneflow, EncoderRNN_oneflow
from utils.utils_oneflow import *
import random
import oneflow as flow
from utils.dataset import prepareData
import argparse


def _parse_args():
    parser = argparse.ArgumentParser("flags for train seq2seq")

    parser.add_argument(
        "--encoder_path",
        type=str,
        default="./saving_model_oneflow/encoder/",
        help="load pretrain encoder_model dir",
    )

    parser.add_argument(
        "--decoder_path",
        type=str,
        default="./saving_model_oneflow/decoder/",
        help="load pretrain decoder_model dir",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="device",
    )

    return parser.parse_args()


# refer to: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def evaluate(
    encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH
):
    with flow.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_Hidden().to(device)

        encoder_outputs = []

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs.append(encoder_output[0])
        if len(encoder_outputs) != max_length:
            for _ in range(max_length - len(encoder_outputs)):
                encoder_outputs.append(flow.zeros((1, 256)).to(device))
        encoder_outputs = flow.cat(encoder_outputs, dim=0)

        decoder_input = flow.tensor([[SOS_token]]).to(device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = flow.zeros((max_length, max_length)).to(device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )            
            decoder_attentions[di] = decoder_attention.squeeze(0).data

            topv, topi = decoder_output.data.topk(1)
            if topi.squeeze().numpy() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(
                    output_lang.index2word[int(topi.squeeze().numpy())]
                )
            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[: di + 1]


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, attentions = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang
        )
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def evaluateAndShowAttention(
    input_sentence, encoder, attn_decoder, input_lang, output_lang
):
    output_words, attentions = evaluate(
        encoder, attn_decoder, input_sentence, input_lang, output_lang
    )
    print("input =", input_sentence)
    print("output =", " ".join(output_words))
    showAttention(input_sentence, output_words, attentions)


def main(args):

    device = args.device
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)
    e = flow.load(args.encoder_path)
    d = flow.load(args.decoder_path)
    encoder = EncoderRNN_oneflow(input_lang.n_words, 256).to(device)
    decoder = AttnDecoderRNN_oneflow(256, output_lang.n_words, dropout_p=0.1).to(device)
    encoder.load_state_dict(e)
    decoder.load_state_dict(d)
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
