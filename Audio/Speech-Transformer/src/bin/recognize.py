#!/usr/bin/env python

import argparse
import json
import oneflow as flow
import kaldi_io
from transformer import Transformer
from utils import add_results_to_json, process_dict
from data import build_LFR_features
from decoder import Decoder
from encoder import Encoder

parser = argparse.ArgumentParser("End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument(
    "--recog-json",
    type=str,
    default="../../egs/aishell/dump/test/deltafalse/data_simple.json",
    help="Filename of recognition data (json)",
)
parser.add_argument(
    "--dict",
    type=str,
    default="../../egs/aishell/data/lang_1char/train_chars.txt",
    help="Dictionary which should include <unk> <sos> <eos>",
)
parser.add_argument(
    "--result-label",
    type=str,
    default="exp/decode_test/data.json",
    help="Filename of result label data (json)",
)
# model
parser.add_argument(
    "--model-path",
    type=str,
    default="exp/temp/final.pth.tar",
    help="Path to model file created by training",
)
# decode
parser.add_argument("--beam-size", default=5, type=int, help="Beam size")
parser.add_argument("--nbest", default=1, type=int, help="Nbest size")
parser.add_argument(
    "--decode-max-len",
    default=0,
    type=int,
    help="Max output length. If ==0 (default), it uses a "
    "end-detect function to automatically find maximum "
    "hypothesis lengths",
)
parser.add_argument(
    "--LFR_m", default=4, type=int, help="Low Frame Rate: number of frames to stack"
)
parser.add_argument(
    "--LFR_n", default=3, type=int, help="Low Frame Rate: number of frames to skip"
)
# encoder
# TODO: automatically infer input dim
parser.add_argument(
    "--d_input", default=80, type=int, help="Dim of encoder input (before LFR)"
)
parser.add_argument(
    "--n_layers_enc", default=6, type=int, help="Number of encoder stacks"
)
parser.add_argument(
    "--n_head", default=8, type=int, help="Number of Multi Head Attention (MHA)"
)
parser.add_argument("--d_k", default=64, type=int, help="Dimension of key")
parser.add_argument("--d_v", default=64, type=int, help="Dimension of value")
parser.add_argument("--d_model", default=512, type=int, help="Dimension of model")
parser.add_argument("--d_inner", default=2048, type=int, help="Dimension of inner")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
parser.add_argument(
    "--pe_maxlen", default=5000, type=int, help="Positional Encoding max len"
)
# decoder
parser.add_argument(
    "--d_word_vec", default=512, type=int, help="Dim of decoder embedding"
)
parser.add_argument(
    "--n_layers_dec", default=6, type=int, help="Number of decoder stacks"
)
parser.add_argument(
    "--tgt_emb_prj_weight_sharing",
    default=1,
    type=int,
    help="share decoder embedding with decoder projection",
)


def recognize(args):
    # model
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    encoder = Encoder(
        args.d_input * args.LFR_m,
        args.n_layers_enc,
        args.n_head,
        args.d_k,
        args.d_v,
        args.d_model,
        args.d_inner,
        dropout=args.dropout,
        pe_maxlen=args.pe_maxlen,
    )
    decoder = Decoder(
        sos_id,
        eos_id,
        vocab_size,
        args.d_word_vec,
        args.n_layers_dec,
        args.n_head,
        args.d_k,
        args.d_v,
        args.d_model,
        args.d_inner,
        dropout=args.dropout,
        tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
        pe_maxlen=args.pe_maxlen,
    )
    model = Transformer(encoder, decoder)
    model.load_state_dict(flow.load(args.model_path))
    device = flow.device("cuda")
    model.eval()
    model.to(device)
    LFR_m = args.LFR_m
    LFR_n = args.LFR_n
    char_list, sos_id, eos_id = process_dict(args.dict)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    # decode each utterance
    new_js = {}
    with flow.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print("(%d/%d) decoding %s" % (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]["input"][0]["feat"])
            input = build_LFR_features(input, LFR_m, LFR_n)
            input = flow.tensor(input).to(dtype=flow.float32)
            input_length = flow.tensor([input.size(0)], dtype=flow.int64)
            input = input.to(device)
            input_length = input_length.to(device)
            nbest_hyps = model.recognize(input, input_length, char_list, args)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)

    with open(args.result_label, "wb") as f:
        f.write(json.dumps({"utts": new_js}, indent=4, sort_keys=True).encode("utf_8"))


args = parser.parse_args()
print(args, flush=True)
recognize(args)
