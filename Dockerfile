FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -y ca-certificates \
         cmake \
         git \
         curl \
         libopenmpi-dev \
         python3-dev \
         zlib1g-dev \
         libgl1-mesa-glx \
	 libglib2.0-0 \
	 libgtk2.0-dev \
         swig && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create --name TradeMaster python=3.9

ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/TradeMaster-NTU/TradeMaster.git /home/TradeMaster
RUN cd /home/TradeMaster
RUN conda update -y conda
RUN conda init bash
RUN echo "conda activate TradeMaster" >> ~/.bashrc
RUN . ~/.bashrc

# Install torch
RUN /opt/conda/envs/TradeMaster/bin/python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install apex
WORKDIR /home
RUN /opt/conda/envs/TradeMaster/bin/python -m pip install packaging
RUN git clone https://github.com/NVIDIA/apex
WORKDIR apex
RUN /opt/conda/envs/TradeMaster/bin/python -m pip install -v --no-cache-dir .

# Install requirements
WORKDIR /home/TradeMaster
RUN /opt/conda/envs/TradeMaster/bin/python -m pip install -r requirements.txt
