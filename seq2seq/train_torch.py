from models.seq_seq_torch import *
from utils.utils_torch import *
import torch.optim as optim
import torch.nn as nn
import time
import random
import os
import torch

# 指导模式概率
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()   # 0时刻的encoder的输入hidden:(1,1,hidden)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)   # encoder的所有输出:(max_len,hidden)

    loss = 0

    for ei in range(input_length):
        # encoder_output:[batch,seq_len,hid]  encoder_hidden:[batch,hid_size]
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)      # 每个词依次进入RNN
        encoder_outputs[ei] = encoder_output[0, 0]      # shape = (hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)      # decoder接收的第一个词为"[SOS]"

    decoder_hidden = encoder_hidden     # decoder的初始hidden

    # ‘指导模式’即：用不用target来作为decoder的输入，不用的话就用decoder自己预测出的来作为下一次输入
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:     # 指导模式
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:                       # 非指导模式
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)



if __name__ == '__main__':
    # pre
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    # training
    print('training')
    hidden_size = 256
    encoder1 = EncoderRNN_torch(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN_torch(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

    # 保存模型
    if not os.path.exists('./trained_model'):
        os.mkdir('./trained_model')

    print('saving model...')
    torch.save(encoder1, './trained_model/encoder.pkl')
    torch.save(attn_decoder1, './trained_model/decoder.pkl')

