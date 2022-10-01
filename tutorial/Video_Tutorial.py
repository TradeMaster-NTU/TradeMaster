import sys

sys.path.append("..")
# data downloading
from data.download_data import *

args = parser.parse_args()
dataconfig = Dataconfig(args)
# load config for DeepScalper and build a trader
from agent.DeepScalper.dqn import *

args = parser.parse_args()
trader = DQN(args)
#train valid and test
trader.train_with_valid()
trader.test()
#visualization
from agent.DeepScalper.visulization import *

pdf_path = trader.result_path + "/result.pdf"
csv_path = trader.result_path + "/result.csv"
plot_result(pd.read_csv(csv_path, index_col=0), pdf_path)
