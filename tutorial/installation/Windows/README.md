# Windows
Step 1: Install [WSL](https://docs.microsoft.com/en-us/windows/wsl/)
- Please follow the steps in this [blog](https://docs.microsoft.com/en-us/windows/wsl/install)

Step 2: Install [Anaconda](https://www.anaconda.com/products/individual)
- Please follow the steps in this [blog](https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-18-04/)

Step 3: Install OpenAI

- Open an ubuntu terminal and type:
   ```
   sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
   ```
Step 4: Install [`TradeMaster`](https://github.com/TradeMaster-NTU/TradeMaster)
- Open a terminal amd type 
  ```
  conda create --name TradeMaster python=3.7.13
  ```
  to install a new conda environment for `TradeMaster`
- Install `TradeMaster`
  ```
  git clone https://github.com/TradeMaster-NTU/TradeMaster.git
  ```
- Open the folder `TradeMaster` and open a terminal under the same position
- Install the dependency of `TradeMaster`, run the command:
   ```
   conda activate TradeMaster
   cd ./requirement
   pip install -r requirements.txt
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```
