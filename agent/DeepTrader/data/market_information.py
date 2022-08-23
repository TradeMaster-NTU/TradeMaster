import pandas as pd
import numpy as np


def make_market_information(df, technical_indicator, store_path):
    #based on the information, calculate the average for technical_indicator to present the market average
    all_dataframe_list = []
    for i in range(df.index.unique()[-1] + 1):
        information = df[df.index == i]
        new_dataframe = []
        for tech in technical_indicator:
            tech_value = np.mean(information[tech])
            new_dataframe.append(tech_value)
        all_dataframe_list.append(new_dataframe)
    new_df = pd.DataFrame(all_dataframe_list, columns=technical_indicator)
    new_df.to_csv(store_path)
    return new_df
