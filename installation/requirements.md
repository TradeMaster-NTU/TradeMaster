# Installation
TradeMaster works on Linux, Windows and MacOS. It requires Python 3.9+, CUDA 11.3+ and PyTorch 1.12+.

__Download and install Miniconda__ from the [official webiste](https://docs.conda.io/en/latest/miniconda.html).
## Ceate a conda environment and activate it

  ```
  conda create --name TradeMaster python=3.9
  conda activate TradeMaster
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  ```
## Apex installation
  ```
  pip install packaging
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir .
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
