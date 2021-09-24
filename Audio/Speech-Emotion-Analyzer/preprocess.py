"""Extracting and saving the features of the audio in the dataset"""
import extract_feats.opensmile as of
import extract_feats.librosa as lf
from utils import parse_opt
import os


config = parse_opt()
os.makedirs(config.checkpoint_path)
os.makedirs(config.feature_path)

if config.feature_method == "o":
    of.get_data(
        config, config.data_path, config.train_feature_path_opensmile, train=True
    )

elif config.feature_method == "l":
    lf.get_data(config, config.data_path, config.train_feature_path_librosa, train=True)
