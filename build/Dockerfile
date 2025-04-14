FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

WORKDIR /root
RUN apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Tokyo apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update -y
RUN apt-get install -y git vim xterm x11-apps python3-pip libopencv-dev 

RUN mkdir /root/il_ws
WORKDIR /root/il_ws
RUN git clone https://github.com/ayakasakurai0213/kuka_il_sim.git -b add_docker

WORKDIR /root/il_ws/kuka_il_sim
RUN ./build/install.sh

WORKDIR /root/il_ws