# Run an interactive session: `sudo docker run -it --rm --gpus all nintorch`
ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

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

RUN wget https://gist.githubusercontent.com/ninfueng/736261c56b094f29704ace23cdace4a4/raw/fb181994fcfe15d598438f7ead454707ed9d7431/.vimrc -P ~

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
