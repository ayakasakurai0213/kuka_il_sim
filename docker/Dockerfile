FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

WORKDIR /root
RUN apt update -y
RUN apt install -y git vim xterm x11-apps python3-pip

RUN pip3 install numpy==1.26.4 \
    matplotlib==3.10.1 \
    pandas==2.2.3 \
    seaborn==0.13.2 \
    ipython==8.34.0 \
    opencv-python==4.11.0.86 \
    pygame==2.6.1 \
    pybullet==3.2.7 \
    gym==0.26.2 \
    h5py==3.13.0 \
    dm_env==1.6 \
    tqdm==4.67.1 \
    diffusers==0.32.2 \
    imageio==2.37.0 \
    einops==0.8.0 \
    termcolor==2.5.0

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118