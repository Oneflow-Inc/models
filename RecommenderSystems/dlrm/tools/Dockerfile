FROM oneflowinc/oneflow:nightly-cuda11.2

RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    apt install default-jdk scala git

    # openjdk-11-jdk 
RUN wget https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz \
    && tar xvf spark-3.2.1-bin-hadoop3.2.tgz \
    && mv spark-3.2.1-bin-hadoop2.7 /opt/spark

ENV SPARK_HOME="/opt/spark"
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

ENV TUNA_PIP_INSTALL=" -i https://pypi.tuna.tsinghua.edu.cn/simple"