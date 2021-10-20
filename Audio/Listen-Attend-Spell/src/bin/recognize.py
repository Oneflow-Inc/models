#!/usr/bin/env python
import argparse
import json

import oneflow as flow
from decoder import Decoder
from encoder import Encoder
from seq2seq import Seq2Seq
from utils import process_dict

import kaldi_io
from seq2seq import Seq2Seq
from utils import add_results_to_json, process_dict


parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--recog_json', type=str, default='../../egs/aishell/dump/dev/deltatrue/data_simple.json',
                    help='Filename of recognition data (json)')
parser.add_argument('--dict', type=str, default='../../egs/aishell/data/lang_1char/train_chars.txt',
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--result_label', type=str, default='exp/decode_test/data.json',
                    help='Filename of result label data (json)')
# model
parser.add_argument('--model_path', type=str, default='exp/temp/final.pth.tar',
                    help='Path to model file created by training')
# decode
parser.add_argument('--beam_size', default=5, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode_max_len', default=0, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')
parser.add_argument('--einput', default=240, type=int,
                    help='Dim of encoder input')
parser.add_argument('--ehidden', default=256, type=int,
                    help='Size of encoder hidden units')
parser.add_argument('--elayer', default=3, type=int,
                    help='Number of encoder layers.')
parser.add_argument('--edropout', default=0.2, type=float,
                    help='Encoder dropout rate')
parser.add_argument('--ebidirectional', default=1, type=int,
                    help='Whether use bidirectional encoder')
parser.add_argument('--etype', default='lstm', type=str,
                    help='Type of encoder RNN')
# attention
parser.add_argument('--atype', default='dot', type=str,
                    help='Type of attention (Only support Dot Product now)')
# decoder
parser.add_argument('--dembed', default=512, type=int,
                    help='Size of decoder embedding')
parser.add_argument('--dhidden', default=512, type=int,
                    help='Size of decoder hidden units. Should be encoder '
                    '(2*) hidden size dependding on bidirection')
parser.add_argument('--dlayer', default=1, type=int,
                    help='Number of decoder layers.')

def recognize(args):
    # model
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    encoder = Encoder(args.einput, args.ehidden, args.elayer,
                      dropout=args.edropout, bidirectional=args.ebidirectional)
    decoder = Decoder(vocab_size, args.dembed, sos_id,
                      eos_id, args.dhidden, args.dlayer,
                      bidirectional_encoder=args.ebidirectional)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(flow.load(args.model_path))
    device = flow.device("cuda")
    model.eval()
    model.to(device)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with flow.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]['input'][0]['feat'])
            input = flow.tensor(input).to(dtype=flow.float32)
            input_length = flow.tensor([input.size(0)], dtype=flow.int64)
            input = input.to(device)
            input_length = input_length.to(device)
            nbest_hyps = model.recognize(input, input_length, char_list, args)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))

args = parser.parse_args()
print(args, flush=True)
recognize(args)
