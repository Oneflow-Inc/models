from utils.utils_torch import *
import random
import torch
from utils.dataset import prepareData




def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateAndShowAttention(input_sentence, encoder, attn_decoder):
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == "__main__":
    # load model
    encoder = torch.load('./saving_model_torch/encoder.pkl')
    attn_decoder = torch.load('./saving_model_torch/decoder.pkl')
    # load dataset
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    # evaluating
    print('evaluating')
    evaluateRandomly(encoder, attn_decoder, pairs)
    # 
    # output_words, attentions = evaluate(encoder, attn_decoder, "je suis trop froid .", input_lang, output_lang)
    # plt.matshow(attentions.numpy())
    # plt.savefig('./attention.jpg')
    # evaluateAndShowAttention("elle a cinq ans de moins que moi .", encoder, attn_decoder)
    #
    # evaluateAndShowAttention("elle est trop petit .", encoder, attn_decoder)
    #
    # evaluateAndShowAttention("je ne crains pas de mourir .", encoder, attn_decoder)
    #
    # evaluateAndShowAttention("c est un jeune directeur plein de talent .", encoder, attn_decoder)