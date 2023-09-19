ARG PYTORCH="2.0.1"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN conda update -n base -c defaults conda
RUN echo "Install Python packages."
RUN pip3 install \
    albumentations \
    seaborn \
    ttach \
    timm \
    cleanlab \
    cleanvision \
    wandb \
    catalyst \
    pytorch_optimizer \
    pybind11 \
    datasets \
    transformers \
    pytest \
    pre-commit \
    torchprofile \
    webdataset \
    "black[jupyter]" \
    isort

RUN echo "Install Ubuntu packages."
RUN \
    apt update && \
    apt upgrade && \
    apt install --no-install-recommends -y \
        wget \
        curl \
        build-essential \
        tar \
        zip \
        htop \
        openssl \
        ninja

# WORKDIR /usr/

# sudo docker run -it --rm --gpus all nintorch
