# -*- coding: utf-8 -*-
# /usr/bin/python2
"""
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
"""


class Hyperparams:
    """Hyper parameters"""

    # pipeline
    prepro = (
        False  # if True, run `python prepro.py` first before running `python train.py`.
    )

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    max_duration = 10.0
    top_db = 15

    # signal processing
    sr = 24000  # Sample rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 512  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 100  # Number of inversion iterations
    preemphasis = 0.97  # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256  # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5  # Reduction factor. Paper => 2, 3, 5
    dropout_rate = 0.5

    # training scheme
    lr = 0.001  # Initial learning rate.
    logdir = "logdir/01"
    sampledir = "samples"
    batch_size = 32
