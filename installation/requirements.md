# Installation
TradeMaster works on Linux, Windows and MacOS for both CPU and GPU, which requires Python 3.9+, CUDA 11.3+ and PyTorch 1.12+. If you have not installed CUDA, please download it from the [official website](https://developer.nvidia.com/cuda-11.3.0-download-archive)

__Download and install Miniconda__ from the [official website](https://docs.conda.io/en/latest/miniconda.html).
## Ceate a conda environment and activate it

  ```
  conda update -n base -c defaults conda
  conda create --name TradeMaster python=3.9
  conda activate TradeMaster
   ```
  
## Pytorch installation
install Pytorch following [official instructions](https://pytorch.org/):

On CPU platforms
  ```
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
  ```
On GPU platforms
  ```
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  ```
## Apex installation
Install Apex following [apex repo](https://github.com/NVIDIA/apex):
  ```
  pip install packaging mkl
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir .
  cd ..
  ```
If on CPU platforms,
  ```
  pip install packaging mkl
  git clone https://github.com/NVIDIA/apex
  cd apex
  git switch python_only
  pip install -v --no-cache-dir .
  cd ..
  ```
## TradeMaster installation  

  ```
  git clone https://github.com/TradeMaster-NTU/TradeMaster.git
  cd TradeMaster
  pip install -r requirements.txt
  ```

##  Verify installation

  ```
  python tools/algorithmic_trading/train.py
  ```
