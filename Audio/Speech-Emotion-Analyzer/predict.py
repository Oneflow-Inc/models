import os
import numpy as np
import oneflow as flow
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import utils
from models import lstm_ser
from models import cnn1d_ser


def predict(config) -> None:
    """
    Predict the emotion of the input audio

    Args:
        confguration items
        audio_path (str): path of input audio
    """

    # utils.play_audio(audio_path)
    if config.feature_method == "o":
        of.get_data(
            config,
            config.audio_path,
            config.predict_feature_path_opensmile,
            train=False,
        )
        test_feature = of.load_feature(
            config, config.predict_feature_path_opensmile, train=False
        )
    elif config.feature_method == "l":
        test_feature = lf.get_data(
            config, config.audio_path, config.predict_feature_path_librosa, train=False
        )

    test_feature = test_feature.reshape(1, test_feature.shape[0], test_feature.shape[1])
    test_feature = flow.tensor(test_feature, dtype=flow.float32, device="cuda")

    n_feats = test_feature.shape[2]

    if config.model == "lstm":
        model = lstm_ser(n_feats, config.rnn_size, len(config.class_labels), 1)
    else:
        model = cnn1d_ser(
            1, config.n_kernels, n_feats, config.hidden_size, len(config.class_labels)
        )
    SER_model = model
    SER_model.to("cuda")

    model_path = os.path.join(config.checkpoint_path, config.checkpoint_name)
    SER_model.load_state_dict(flow.load(model_path))
    flow.no_grad()

    logits = SER_model(test_feature)
    result = np.argmax(logits.numpy(),)
    print("Recognition:", config.class_labels[int(result)])

    result_prob = flow.softmax(logits, dim=1)
    utils.radar(result_prob.numpy().squeeze(), config.class_labels)


config = utils.parse_opt()

predict(config)
