# Run an interactive session: `docker run -it --rm --gpus all nintorch`
ARG PYTORCH="2.5.1"
ARG CUDA="12.4"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
RUN apt update -y && \
    apt upgrade -y && \
    apt install --no-install-recommends -y \
        wget \
        curl \
        build-essential \
        ninja-build \
        tar \
        zip \
        unzip \
        git \
        htop \
        vim \
        tmux

RUN conda update -n base -c defaults conda
RUN pip3 install --no-cache \
    albumentations \
    seaborn \
    timm \
    wandb \
    pybind11 \
    datasets \
    transformers \
    pytest \
    pre-commit \
    torchprofile \
    git+https://github.com/ninfueng/nincore.git@master \
    git+https://github.com/ninfueng/nintorch.git@master

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
