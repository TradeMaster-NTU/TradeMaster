## Tutorial

### 0. 下載Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
source ~/.bashrc
conda --version
會顯示像這樣：

```bash
conda 24.3.0
```

### 1. 啟動腳本

cd NYCU2025DLfinal-TradeMaster
bash NYCU-DL-Final/setup_env.sh

### 2. 接著會提示你啟動環境
conda activate TradeMaster
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118


### 3. 測試環境
cd tools/portfolio_management
python train_eiie.py

### 4. 跑得起來就沒問題啦

