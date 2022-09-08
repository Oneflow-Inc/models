apt-get update
apt-get install -y default-jdk
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -r /workspace/test/requirements.txt
python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112
python3 -m oneflow --doctor
