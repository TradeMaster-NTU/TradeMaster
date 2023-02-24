# Installation using Requirements
## Build Environment using [`Anaconda`]
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
   pip install -r requirements.txt
   ```

##  Test Installation
  ```
  python tools/algorithmic_trading/train.py
  ```