from models.seq_seq_onwflow import AttnDecoderRNN_oneflow, EncoderRNN_oneflow
from utils.utils_oneflow import *
import random
import oneflow.experimental as flow
from utils.dataset import prepareData


# refer to: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with flow.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_Hidden()
        
        encoder_outputs = []
 
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs.append(encoder_output[0])
        if len(encoder_outputs)!=max_length:
            for _ in range(max_length-len(encoder_outputs)):
                encoder_outputs.append(flow.zeros((1,256)).to(device))
        encoder_outputs = flow.cat(encoder_outputs,dim=0)

        decoder_input = flow.tensor([[SOS_token]]).to(device)

        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = flow.zeros((max_length, max_length)).to(device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.squeeze().numpy() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[int(topi.squeeze().numpy())])

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
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    e = flow.load('./saving_model_oneflow/encoder/')
    d = flow.load('./saving_model_oneflow/decoder/')
    encoder = EncoderRNN_oneflow(input_lang.n_words, 256).to(device)
    decoder = AttnDecoderRNN_oneflow(256, output_lang.n_words, dropout_p=0.1).to(device)
    encoder.load_state_dict(e)
    decoder.load_state_dict(d)
    evaluateRandomly(encoder, decoder, pairs)
    output_words, attentions = evaluate(encoder, decoder, "je suis trop froid .", input_lang, output_lang)
