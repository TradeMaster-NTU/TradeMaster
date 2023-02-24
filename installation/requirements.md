# Installation
TradeMaster works on Linux, Windows and MacOS. It requires Python x.x+, CUDA x.x+ and PyTorch x.x+.
__Download and install Miniconda__ from the [official webiste](https://docs.conda.io/en/latest/miniconda.html)
## Ceate a conda environment and activate it

  ```
  conda create --name TradeMaster python=3.7.13
  conda activate TradeMaster
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
