import oneflow as flow
import oneflow.nn.functional as F
import yaml
import pickle
from model import AE
from utils import *
from functools import reduce
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms


class Inferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, "rb") as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f"Load model from {self.args.model}")
        self.model.load_state_dict(flow.load(f"{self.args.model}"))
        return

    def build_model(self):
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config["data_loader"]["frame_size"]
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def inference_from_path(self):
        src_mel, _ = get_spectrograms(self.args.source)
        tar_mel, _ = get_spectrograms(self.args.target)
        src_mel = flow.tensor(self.normalize(src_mel)).cuda()
        tar_mel = flow.tensor(self.normalize(tar_mel)).cuda()
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        self.write_wav_to_file(conv_wav, self.args.output)
        return


parser = ArgumentParser()
parser.add_argument("-attr", "-a", help="attr file path")
parser.add_argument("-config", "-c", help="config file path")
parser.add_argument("-model", "-m", help="model path")
parser.add_argument("-source", "-s", help="source wav path")
parser.add_argument("-target", "-t", help="target wav path")
parser.add_argument("-output", "-o", help="output wav path")
parser.add_argument("-sample_rate", "-sr", help="sample rate", default=24000, type=int)
args = parser.parse_args()
# load config file
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
inferencer = Inferencer(config=config, args=args)
inferencer.inference_from_path()
