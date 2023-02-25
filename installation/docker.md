# Installation using Docker
## Build the docker image from [dockerfile](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/docker/Dockerfile) make sure docker service is started
- Install `TradeMaster`
  ```
   git clone https://github.com/TradeMaster-NTU/TradeMaster.git
   cd TradeMaster
  ```
- Create image from the project docker file.

  ```
  docker build -t tardemaster:1.0.0 .
  ```

  It will take a while before the image is built.

##  Test whether the image is installed correctly

- Open the terminal in the project position and type

  ```
  docker images
  ```

  It should shows 

  ```
  REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
  trademaster   1.0.0       02801f755797   4 minutes ago   15GB 
  ```

- Create a container and run an experiment to see whether the installation is successful

  ```
  docker run -it --name trademaster trademaster:1.0.0 /bin/bash
  python tools/algorithmic_trading/train.py
  ```
