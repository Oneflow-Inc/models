FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt update &&  DEBIAN_FRONTEND=noninteractive apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool libjemalloc2 google-perftools
RUN python3 -m pip install --pre oneflow -f https://staging.oneflow.info/commit/65fd4d99ad435baefe500e6ea9aaebcbb36c905e/cu116 psutil petastorm
RUN git clone --branch dlrm_benchmark_test --depth 1  https://github.com/Oneflow-Inc/models.git /models
RUN cd /models/RecommenderSystems/dlrm
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libjemalloc.so.2
