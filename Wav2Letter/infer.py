import os
import argparse

import oneflow as flow

from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
from Wav2Letter.decoder import GreedyDecoder


def get_args():
    parser = argparse.ArgumentParser(
        """Wav2Letter train"""
    )
    parser.add_argument("--mfcc_features", type=int, default=13)
    parser.add_argument("--datasets_path", type=str, default="speech_data")
    parser.add_argument("--output_path", type=str, default="save_models")

    args = parser.parse_args()
    return args

def infer(opt):
    mfcc_features = opt.mfcc_features
    datasets_path = opt.datasets_path
    models_path = opt.output_path

    # load saved numpy arrays for google speech command
    gs = GoogleSpeechCommand()
    _inputs, _targets = gs.load_vectors(datasets_path)
    grapheme_count = gs.intencode.grapheme_count

    inputs = flow.Tensor(_inputs).to('cuda')
    targets = flow.Tensor(_targets, dtype=flow.int).to('cuda')

    model = Wav2Letter(mfcc_features, grapheme_count)
    model.to('cuda')
    model.load_state_dict(flow.load(os.path.join(models_path, "model.pth")))

    decoder = GreedyDecoder()

    inputs = inputs.transpose(1, 2)

    sample = inputs[-1000:]
    sample_target = targets[-1000:]
    
    log_probs = model(sample)
    output = decoder.decode(log_probs)

    pred_strings, output = decoder.convert_to_strings(output)
    sample_target_strings = decoder.convert_to_strings(sample_target, remove_repetitions=False, return_offsets=False)
    wer = decoder.wer(sample_target_strings, pred_strings)

    print("wer", wer)


if __name__ == "__main__":
    opt = get_args()
    infer(opt)
