model="lstm" # choose between rnn and lstm

if [ ! -d "data/" ]; then
    wget https://download.pytorch.org/tutorial/data.zip
    unzip data.zip
fi

echo "begin ${model} speed comparison demo"

python3 compare_oneflow_and_pytorch_${model}_speed.py

echo "begin ${model} training demo"

python3 train_${model}_oneflow.py