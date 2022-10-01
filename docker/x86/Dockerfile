FROM ubuntu:20.04 as base
ARG PYTHON_VERSION=3.7.13

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -y ca-certificates \
         cmake \
         git \
         curl \
         libopenmpi-dev \
         python3-dev \
         zlib1g-dev \
         libgl1-mesa-glx \
         swig && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create --name TradeMaster python=$PYTHON_VERSION

ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/TradeMaster-NTU/TradeMaster.git /home/TradeMaster
RUN cd  /home/TradeMaster && \
        conda init bash && . ~/.bashrc && \
        conda activate TradeMaster && \
        pip install -r requirement/requirements.txt && \
        conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN echo "conda activate TradeMaster" >> ~/.bashrc
WORKDIR /home/TradeMaster
