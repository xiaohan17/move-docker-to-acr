# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8        (apt)
# pytorch       1.6 cuda   (pip)
# ==================================================================
FROM nvidia/cuda:11.6.2-base-ubuntu18.04
ENV LANG C.UTF-8
WORKDIR /app
COPY src /app/src
COPY ./multiple_layer_perceptron.py /app
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --user --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        apt-utils \
        ca-certificates \
        wget \
        libssl-dev libglib2.0-dev libsm6 libxext6 libxrender-dev && \
# ==================================================================
# python
# ------------------------------------------------------------------
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        && \
    wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    pip install -U pip && \
    pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    $PIP_INSTALL numpy==1.18.5 opencv-contrib-python==4.1.2.30 pillow==7.2.0 tqdm==4.47.0 && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
        --pre pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -f \
        https://download.pytorch.org/whl/cu116 \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
ENTRYPOINT ["python", "multiple_layer_perceptron.py"]
CMD ["--num_epochs", "20"]

