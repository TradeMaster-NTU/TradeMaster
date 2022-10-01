from matplotlib import pyplot as plt
import pandas as pd


def plot_result(test_result, plot_path):
    x = range(len(test_result))
    y = test_result["total assets"].tolist()
    plt.plot(x, y)
    plt.xlabel("the number of trading timestamp")
    plt.ylabel("the total asset")
    plt.grid()
    plt.show()
    plt.savefig(plot_path)


if __name__ == "__main__":
    plot_result(pd.read_csv("result/AT/test_result/result.csv", index_col=0),
                "result/AT/test_result/result.pdf")
