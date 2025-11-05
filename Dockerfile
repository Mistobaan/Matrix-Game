# see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html
# this is the last image that supports cuda 12.x and pytorch official is built with that CUDA version
# pytorch has to be 2.8 too so 
FROM nvcr.io/nvidia/pytorch:25.05-py3

RUN pip install \
    torchao==0.13.0 # see https://github.com/pytorch/ao/issues/2919 \
    diffusers==0.31.0
# pip install numpy==2.2.6
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128