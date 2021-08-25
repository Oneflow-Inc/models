set -aux

DATA_PATH="speech_data"

if [ ! -d “$DATA_PATH” ]; then
    wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    mkdir $DATA_PATH && mkdir $DATA_PATH/speech_commands_v0.01
    tar -xvf speech_commands_v0.01.tar -C $DATA_PATH/speech_commands_v0.01
    rm -fr speech_commands_v0.01.tar
fi
echo "Data download success!"

python Wav2Letter/data.py
echo "Data proprecessed!"

python train.py
