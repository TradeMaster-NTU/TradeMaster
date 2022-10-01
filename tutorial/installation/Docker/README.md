# Introduction to Docker
Docker is a set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers. 

# Installation 
Step 1: Install [Docker](https://docs.docker.com/)
- Please follow the steps in this [blog](https://docs.docker.com/engine/install/)
- To check whether the Docker has been installed properly, type `docker version`, it should show:
  ```
  Client:
   Cloud integration: v1.0.29
   Version:           20.10.17
   API version:       1.41
   Go version:        go1.17.11
   Git commit:        100c701
   Built:             Mon Jun  6 23:09:02 2022
   OS/Arch:           windows/amd64
   Context:           default
   Experimental:      true

  Server: Docker Desktop 4.12.0 (85629)
   Engine:
    Version:          20.10.17
    API version:      1.41 (minimum version 1.12)
    Go version:       go1.17.11
    Git commit:       a89b842
    Built:            Mon Jun  6 23:01:23 2022
    OS/Arch:          linux/amd64
    Experimental:     false
   containerd:
    Version:          1.6.8
    GitCommit:        9cd3357b7fd7218e4aec3eae239db1f68a5a6ec6
   runc:
    Version:          1.1.4
    GitCommit:        v1.1.4-0-g5fd4c4d
   docker-init:
    Version:          0.19.0
    GitCommit:        de40ad0
  ```

Step 2: Build the docker image from [dockerfile](https://github.com/TradeMaster-NTU/TradeMaster/blob/main/docker/Dockerfile)
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

Step 3: Test whether the image is installed correctly

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
  python experiment/AT/DeepScalper/experiment.py
  ```
  