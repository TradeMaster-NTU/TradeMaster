# Installation with Docker
TradeMaster works on Linux, Windows and MacOS. It requires Python 3.9+, CUDA 11.3+ and PyTorch 1.12+.

__Download and install Docker__ from the [official webiste](https://docs.docker.com/engine/install/).

## Build the docker image based on docker file
- Install `TradeMaster`
  ```
   git clone https://github.com/TradeMaster-NTU/TradeMaster.git
   cd TradeMaster
  ```
- Create image from the project docker file

  ```
  docker build -t tardemaster:1.0.0 .
  ```


##  Run TradeMaster with docker

- Create a container and run an experiment to verify the installation

  ```
  docker images
  docker run -it --name trademaster trademaster:1.0.0 /bin/bash
  python tools/algorithmic_trading/train.py
  ```
