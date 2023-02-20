import os.path as osp
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':

    # test dataset
    pytest.main(["{}".format(osp.join(ROOT, "unit_testing", "test_datasets", "test_algorithmic_trading.py"))])
