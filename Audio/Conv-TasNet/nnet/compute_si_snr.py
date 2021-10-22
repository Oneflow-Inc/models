#!/usr/bin/env python

"""
Compute SI-SDR as the evaluation metric
"""

import argparse
from tqdm import tqdm
from collections import defaultdict
from libs.metric import si_snr, permute_si_snr
from libs.audio import WaveReader, Reader


class SpeakersReader(object):
    def __init__(self, scps):
        split_scps = scps.split(",")
        if len(split_scps) == 1:
            raise RuntimeError(
                "Construct SpeakersReader need more than one script, got {}".
                format(scps))
        self.readers = [WaveReader(scp) for scp in split_scps]

    def __len__(self):
        first_reader = self.readers[0]
        return len(first_reader)

    def __getitem__(self, key):
        return [reader[key] for reader in self.readers]

    def __iter__(self):
        first_reader = self.readers[0]
        for key in first_reader.index_keys:
            yield key, self[key]


class Report(object):
    def __init__(self, spk2gender=None):
        self.s2g = Reader(spk2gender) if spk2gender else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)

    def add(self, key, val):
        gender = "NG"
        if self.s2g:
            gender = self.s2g[key]
        self.snr[gender] += val
        self.cnt[gender] += 1

    def report(self):
        print("SI-SDR(dB) Report: ")
        for gender in self.snr:
            tot_snrs = self.snr[gender]
            num_utts = self.cnt[gender]
            print("{}: {:d}/{:.3f}".format(gender, num_utts,
                                           tot_snrs / num_utts))


def run(args):
    single_speaker = len(args.sep_scp.split(",")) == 1
    reporter = Report(args.spk2gender)

    if single_speaker:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
        for key, sep in tqdm(sep_reader):
            ref = ref_reader[key]
            if sep.size != ref.size:
                end = min(sep.size, ref.size)
                sep = sep[:end]
                ref = ref[:end]
            snr = si_snr(sep, ref)
            reporter.add(key, snr)
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)
        for key, sep_list in tqdm(sep_reader):
            ref_list = ref_reader[key]
            if sep_list[0].size != ref_list[0].size:
                end = min(sep_list[0].size, ref_list[0].size)
                sep_list = [s[:end] for s in sep_list]
                ref_list = [s[:end] for s in ref_list]
            snr = permute_si_snr(sep_list, ref_list)
            reporter.add(key, snr)
    reporter.report()

parser = argparse.ArgumentParser(
    description=
    "Command to compute SI-SDR, as metric of the separation quality",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--sep_scp",
    type=str,
    help="Separated speech scripts, waiting for measure"
    "(support multi-speaker, egs: spk1.scp,spk2.scp)")
parser.add_argument(
    "--ref_scp",
    type=str,
    help="Reference speech scripts, as ground truth for"
    " SI-SDR computation")
parser.add_argument(
    "--spk2gender",
    type=str,
    default="",
    help="If assigned, report results per gender")
args = parser.parse_args()
run(args)