import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

pretrained = dict(
    sarl_encoder = os.path.join(ROOT, "trademaster", "pretrained", "sarl_encoder", "LSTM.pth"),
    investor_imitator = os.path.join(ROOT, "trademaster", "pretrained", "logic_discriptor"),
)