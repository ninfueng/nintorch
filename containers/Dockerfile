# Run an interactive session: `sudo docker run -it --rm --gpus all nintorch`
ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

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

RUN \
    apt update -y && \
    apt upgrade -y && \
    apt install --no-install-recommends -y \
        wget \
        curl \
        build-essential \
        ninja-build \
        tar \
        zip \
        git \
        htop \
        vim \
        tmux

RUN wget https://gist.githubusercontent.com/ninfueng/736261c56b094f29704ace23cdace4a4/raw/fb181994fcfe15d598438f7ead454707ed9d7431/.vimrc -P ~

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]