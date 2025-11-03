# see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html
FROM nvcr.io/nvidia/pytorch:25.08-py3

RUN pip install \
    flash_attn==2.8.3 \
    torchao==0.13.0 # see https://github.com/pytorch/ao/issues/2919 \
    diffusers==0.31.0
# pip install numpy==2.2.6
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128