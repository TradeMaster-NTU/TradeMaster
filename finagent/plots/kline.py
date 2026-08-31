import pandas as pd
import pandas_ta_classic as ta
import numpy as np

from snapshot_selenium import snapshot as driver
from pyecharts import options as opts
from pyecharts.charts import Kline, Bar, Line, Grid
from pyecharts.render import make_snapshot
import os

from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = ""

def cal_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def cal_macd(data, short_window, long_window):
    short_ema = cal_ema(data['Close'], short_window)
    long_ema = cal_ema(data['Close'], long_window)
    macd_line = short_ema - long_ema
    return macd_line

def plot_kline(df,
               title,
               save_path,
               now_date,
               width = 3.5,
               opacity = 0.8,
               path= None,
               mode = "train"):

    if not mode == "train":
        df = df[df.index <= now_date]

    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'timestamp': 'Date'})

    format = "%Y-%m-%d"
    date = df.index.strftime(format).tolist()
    values = df[["Open", "Close", "Low", "High"]].values.tolist()
    volumes = [[i, row[1]["Volume"], 1 if row[1]["Open"] > row[1]["Close"] else -1] for i, row in enumerate(df.iterrows())]

    # Calculate the indicators
    # df['sma_3'] = ta.sma(df["Close"], length=3)
    df['sma_5'] = ta.sma(df["Close"], length=5)
    # df['sma_7'] = ta.sma(df["Close"], length=7)
    bbands = ta.bbands(df["Close"], length=5)
    df['bbl'] = bbands.iloc[:, 0]
    df['bbu'] = bbands.iloc[:, 2]
    df['bbp'] = bbands.iloc[:, 4]

    macd = cal_macd(df, 7, 14)
    max_macd = max(abs(x) for x in macd)
    max_volume = max(df["Volume"])
    adj_macd = [number * max_volume / max_macd for number in macd]

    kline = (
        Kline()
        .add_xaxis(xaxis_data=date)
        .add_yaxis(
            series_name=title,
            y_axis=values,
            itemstyle_opts=opts.ItemStyleOpts(color="#00da3c", color0="#ec0000", border_color="#00da3c", border_color0="#ec0000"),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(
                        coord=[now_date, df.loc[now_date, 'High']],
                        value=now_date,
                        symbol_size=100,
                        symbol="pin",
                        itemstyle_opts=opts.ItemStyleOpts(color="grey"),
                    )
                ]
            )
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(
                is_show=True
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                series_index=4,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#ec0000"},
                    {"value": -1, "color": "#00da3c"},
                ],
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="lineX",
            ),
        )
    )

    line = (
        Line()
        .add_xaxis(xaxis_data=date)
        # .add_yaxis(
        #     series_name="MA3",
        #     y_axis=df['sma_3'].values.tolist(),
        #     is_smooth=False,
        #     is_hover_animation=False,
        #     linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
        #     label_opts=opts.LabelOpts(is_show=False),
        # )
        .add_yaxis(
            series_name="MA5",
            y_axis=df['sma_5'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        # .add_yaxis(
        #     series_name="MA7",
        #     y_axis=df['sma_7'].values.tolist(),
        #     is_smooth=False,
        #     is_hover_animation=False,
        #     linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
        #     label_opts=opts.LabelOpts(is_show=False),
        # )
        .add_yaxis(
            series_name="BBL",
            y_axis=df['bbl'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="BBU",
            y_axis=df['bbu'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
    )

    # bar = (
    #     Bar()
    #     .add_xaxis(xaxis_data=date)
    #     .add_yaxis(
    #         series_name="Volume",
    #         y_axis=volumes,
    #         xaxis_index=1,
    #         yaxis_index=1,
    #         label_opts=opts.LabelOpts(is_show=False),
    #     )
    #     .set_global_opts(
    #         legend_opts=opts.LegendOpts(
    #             is_show=True,
    #             orient="horizontal", pos_top="80%"
    #         ),
    #         xaxis_opts=opts.AxisOpts(
    #             type_="category",
    #             is_scale=True,
    #             grid_index=1,
    #             boundary_gap=False,
    #             axisline_opts=opts.AxisLineOpts(is_on_zero=False),
    #             axistick_opts=opts.AxisTickOpts(is_show=False),
    #             splitline_opts=opts.SplitLineOpts(is_show=False),
    #             axislabel_opts=opts.LabelOpts(is_show=False),
    #             split_number=20,
    #             min_="dataMin",
    #             max_="dataMax",
    #         ),
    #         yaxis_opts=opts.AxisOpts(
    #             grid_index=1,
    #             is_scale=True,
    #             split_number=2,
    #             axislabel_opts=opts.LabelOpts(is_show=False),
    #             axisline_opts=opts.AxisLineOpts(is_show=False),
    #             axistick_opts=opts.AxisTickOpts(is_show=False),
    #             splitline_opts=opts.SplitLineOpts(is_show=False),
    #         ),
    #     )
    # )

    # bar_line = (
    #     Line()
    #     .add_xaxis(xaxis_data=date)
    #     .add_yaxis(
    #         series_name="MACD",
    #         y_axis=adj_macd,
    #         is_smooth=False,
    #         is_hover_animation=False,
    #         linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
    #         label_opts=opts.LabelOpts(is_show=False),
    #     )
    #     .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
    # )

    # Kline And Line
    overlap_kline_line = kline.overlap(line)

    # Bar And Bar_Line
    # overlap_bar = bar.overlap(bar_line)

    # Grid Overlap + Bar
    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="600px",
            height="400px",
            animation_opts=opts.AnimationOpts(animation=False),
            bg_color="white",
        )
    )
    grid_chart.add(
        overlap_kline_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_top= "20%", pos_bottom="10%", height="70%"),
    )
    # grid_chart.add(
    #     overlap_bar,
    #     grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="63%", height="16%"),
    # )

    if path is None:
        path = os.path.join(os.path.dirname(save_path), 'kline.html')

    make_snapshot(driver, grid_chart.render(path=path), save_path, is_remove_html=True)
