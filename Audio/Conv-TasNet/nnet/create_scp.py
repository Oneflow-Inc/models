#!/usr/bin/env python
"""
@Yingzhao <yinger_z@126.com>
"""

import argparse
import os

parser = argparse.ArgumentParser("Command to create scp file")
parser.add_argument("--dataPath", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--scp_name", type=str, required=True)


def GenerateScp(args):
    train_s2 = os.path.join(args.dataPath, args.data)
    train_s2_scp = os.path.join(args.dataPath, args.scp_name)
    tr_mix = open(train_s2_scp, 'w')
    for root, dirs, files in os.walk(train_s2):
        files.sort()
        for file in files:
            tr_mix.write(file + " " + root + '/' + file)
            tr_mix.write('\n')

args = parser.parse_args()
GenerateScp(args)
