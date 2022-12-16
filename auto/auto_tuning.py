import argparse
import sys
from EIIE import EIIE_tuning
from DeepScalper import DeepScalper_tuning
from IMIT import IMIT_tuning


sys.path.append(".")
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm",
                    type=str,
                    default="DeepScalper",
                    help="the name of algorithm for hyperparameter tuning")
args = parser.parse_args()

if __name__ == '__main__':
    if args.algorithm == "EIIE":
        EIIE_tuning()
    elif args.algorithm == "DeepScalper":
        DeepScalper-tuning()
    elif args.algorithm == "IMIT":
        IMIT_tuning()