FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /build

RUN apt-get update && apt-get install -y \
    git 

RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention/hopper \
    && git checkout -b 02541ac \
    && MAX_JOBS=32 python3 setup.py install --no-build-isolation
