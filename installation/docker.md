# Installation using Docker
## Build the docker image from [dockerfile](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/docker/Dockerfile)
- Install `TradeMaster`
  ```
   git clone https://github.com/TradeMaster-NTU/TradeMaster.git
  ```
- Create image from the project docker file.

  If your chip is arm-architectured, open terminal or cmd in the position of the project and type
  ```
  cd ./docker/arm
  docker build -t="trademaster:0.1" .
  ```
  If you chip is x86-architectured, open terminal or cmd in the position of the project and type
  ```
  cd ./docker/x86
  docker build -t="trademaster:0.1" .
  ```
  It will take a while before the image is built.

##  Test whether the image is installed correctly

- Open the terminal in the project position and type
  ```
  docker image ls
  ```
  It should shows 
  ```
  REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
  trademaster   0.1       02801f755797   4 minutes ago   15GB 
  ```
- Create a container and run an experiment to see whether the installation is successful
  ```
  docker run -it trademaster:0.1
  python tools/algorithmic_trading/train.py
  ```