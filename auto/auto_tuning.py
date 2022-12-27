import sys
sys.path.append(".")

import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",
                    type=str,
                    default="EIIE",
                    help="the name of algorithm for hyperparameter tuning")
    args = parser.parse_args()

    if args.algorithm == "EIIE": 
        from EIIE import EIIE_tuning
        EIIE_tuning()
    elif args.algorithm == "DeepScalper": 
        from DeepScalper import DeepScalper_tuning       
        DeepScalper_tuning()
    elif args.algorithm == "IMIT":
        from IMIT import IMIT_tuning
        IMIT_tuning()
