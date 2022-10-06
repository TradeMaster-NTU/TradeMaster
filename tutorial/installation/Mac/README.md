# MAC OS
Step 1: Install [`Anaconda`](https://www.anaconda.com/products/individual)

- Follow Anaconda’s instruction: [`macOS graphical install`](https://docs.anaconda.com/anaconda/install/mac-os/), to install the newest version of Anaconda.

- Open your terminal and type: `which python`, it should show:
  ```
  /Users/your_user_name/opt/anaconda3/bin/python
  ```
  It means that your Python interpreter path has been pinned to Anaconda’s python version. If it shows something like this:
  ```
  /Users/your_user_name/opt/anaconda3/bin/python
  ```
  It means that you still use the default python path, you either fix it and pin it to the anaconda path (try this [blog](https://towardsdatascience.com/how-to-successfully-install-anaconda-on-a-mac-and-actually-get-it-to-work-53ce18025f97)), or you can use Anaconda Navigator to open a terminal manually.

Step 2: Install [`Homebrew`](https://brew.sh/)

- Open a terminal and make sure that you have installed Anaconda.
- Install Homebrew:
  ```
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
Step 3: Install [`OpenAI`](https://github.com/openai/baselines)

Installation of system packages on Mac requires Homebrew. With Homebrew installed, run the following in your terminal:
```
brew install cmake openmpi
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
   conda install pytorch torchvision torchaudio -c pytorch
   ```
