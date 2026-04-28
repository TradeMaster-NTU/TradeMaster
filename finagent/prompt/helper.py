import json
from typing import Dict, Any
from bs4 import BeautifulSoup,Tag
import pandas as pd
from copy import deepcopy
import math
from finagent.memory import MemoryInterface
from finagent.provider import EmbeddingProvider
from finagent.query import DiverseQuery, extract_query_type
from finagent.tools import StrategyAgents
from finagent.asset import ASSET
import os
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
import warnings
from finagent.prompt.sentiment import format_social_sentiment_text
def str2html(doc: str) -> BeautifulSoup:
    doc = BeautifulSoup(doc, 'html.parser')
    return doc

def text_replace(text: str, params: Dict[str, Any]):
    keys = params.keys()

    # find all keys in text
    exiting_keys = []
    for key in keys:
        if f"$${key}$$" in text:
            exiting_keys.append(key)

    # get all values of exiting keys
    exiting_values = [params[key] for key in exiting_keys]

    # if there is None in exiting_values, return None
    if None in exiting_values:
        return None
    else:
        for key in exiting_keys:
            if isinstance(params[key], list) or isinstance(params[key], dict):
                text = text.replace(f"$${key}$$", json.dumps(params[key], indent=4))
            else:
                text = text.replace(f"$${key}$$", str(params[key]))
    return text


def content_replace(content: str):
    maps = {
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&#10;": "\n",
        "&#13;": "\r",
        "&#9;": "\t",
        "&nbsp;": " ",
    }
    for key in maps.keys():
        content = content.replace(key, maps[key])
    return content

def generate_prompt_html(params: Dict[str, Any], template: str):
    template = str2html(template)

    # replace tags by iframe tags
    iframes = template.find_all("iframe")
    for iframe in iframes:
        iframe_name = iframe["name"]

        if not ASSET.check_module(iframe_name):
            raise Exception(f"Module {iframe_name} not found in ASSET[\"modules\"]")

        new_iframe = ASSET.get_module(iframe_name)
        iframe.replace_with(new_iframe)

    p_tags = template.find_all("p", class_="placeholder")

    for p_tag in p_tags:
        text = text_replace(p_tag.text, params)

        if text is None:
            continue
        else:
            p_tag.string = text

        parts = p_tag.string.split('\n')
        new_content = Tag(name='p')
        new_content.attrs = p_tag.attrs
        new_content.append(parts[0])
        for part in parts[1:]:
            new_content.append(Tag(name='br', can_be_empty_element=True))
            new_content.append(part)
        p_tag.replace_with(new_content)

    img_tags = template.find_all("img")
    for img_tag in img_tags:
        img_tag["src"] = params[f"{img_tag['src'].replace('$$', '')}"]

    return template

def prepared_tools_params(state: Dict,
                          info: Dict,
                          params: Dict,
                          memory: MemoryInterface = None,
                          provider: EmbeddingProvider = None,
                          diverse_query: DiverseQuery = None,
                          strategy_agents: StrategyAgents = None,
                          cfg = None,
                          mode= "valid"
                          ):

    price = deepcopy(state["price"])
    news = deepcopy(state["news"])
    guidance = deepcopy(state["guidance"])
    sentiment = deepcopy(state["sentiment"])
    economic = deepcopy(state["economic"])

    res_params = deepcopy(params)

    date = info["date"]

    # prepare guidance
    guidance_text = "There is no guidance today.\n"
    if guidance is not None:
        guidance = guidance[guidance.index == date]
        if len(guidance) == 0:
            guidance_text = "There is no guidance today.\n"
        else:
            guidance_list = []

            for row in guidance.iterrows():
                row = row[1]

                title = row["title"]
                summary = row["text"]
                sentiment = row["sentiment"]

                guidance_text =  f"Headline: {title}\n" + \
                                 f"Content: {summary}\n" + \
                                 f"Sentiment: {sentiment}\n"

                guidance_list.append(guidance_text)

            if len(guidance_list) == 0:
                guidance_text = "There is no guidance today.\n"
            else:
                guidance_text = "\n".join(guidance_list)

    # prepare sentiment
    sentiment_text = format_social_sentiment_text(sentiment=sentiment, date=date)

    # prepare stategies

    strategy_name ={1:"MACD",2:"KDJ & RSI",3:"Stochastic Bollinger",4:"Mean Reversion"}
    # TODO: add stategies
    strategy_text_list=[]
    trading_record_text_list=[]
    for i in range(4):
        strategy_number = i + 1
        data = price[price.index <= date]
        # Do not use best params for now

        # Read the best parameters from training
        if cfg.tool_use_best_params and mode=="valid":
            print(f"Using best params for strategy {strategy_number}")
            best_params_save_path = os.path.join(ROOT, cfg.tool_params_dir, cfg.selected_asset, str(strategy_number),
                                                 "exp001","trained",
                                                 "best_params.json")
            trading_record_save_path = os.path.join(ROOT, cfg.tool_params_dir, cfg.selected_asset, str(strategy_number),
                                                     "exp001","trained",
                                                     "best_result.json")
            try:
                with open(best_params_save_path, "r") as f:
                    best_params = json.load(f)
            except:
                best_params = None
                exit("No best params found, please train the model first")
            strategy_signals, explanations = strategy_agents.wrapper(strategy_number,data, best_params)
        else:
            print("using default params for strategy")
            strategy_signals, explanations = strategy_agents.wrapper(strategy_number,data)
            trading_record_save_path = os.path.join(ROOT, cfg.tool_params_dir, cfg.selected_asset, str(strategy_number),
                                                     "exp001","default",
                                                     "trading_record.json")
        strategy_signals, explanations = strategy_agents.wrapper(strategy_number, data)
        last_signal = strategy_signals[-1]
        last_explanation = explanations[-1]
        strategy_text = f"Strategy: {strategy_name[strategy_number]}\n" + \
                        f"Decision: {last_signal}\n" + \
                        f"Reasoning: {last_explanation}\n"
        strategy_text_list.append(strategy_text)

        # get strategy trading record
        try:
            with open(trading_record_save_path, "r") as f:
                trading_record = json.load(f)
        except:
            trading_record = None
            warnings.warn("No trading record found, please run the benchmark first")

        arr=trading_record["ARR"]
        sr=trading_record["SR"]
        mdd=trading_record["MDD"]
        trading_record=f"Annualized Return Rate: {arr}\n" + \
                        f"Sharpe Ratio: {sr}\n" + \
                        f"Maximum Drawdown: {mdd}\n"
        trading_record_text_list.append(trading_record)

    # get the trading record of buy and hold (strategy 0)
    strategy_number = 0
    trading_record_save_path = os.path.join(ROOT, cfg.tool_params_dir, cfg.selected_asset, str(strategy_number),
                                            "exp001", "default",
                                            "best_result.json")
    try:
        with open(trading_record_save_path, "r") as f:
            trading_record = json.load(f)
    except:
        trading_record = None
        warnings.warn("No trading record found, please run the benchmark first")
    arr = trading_record["ARR"]
    sr = trading_record["SR"]
    mdd = trading_record["MDD"]
    buy_and_hold_record = f"Annualized Return Rate: {arr}\n" + \
                     f"Sharpe Ratio: {sr}\n" + \
                     f"Maximum Drawdown: {mdd}\n"


    # strategy1_macd_signals, strategy1_macd_explanations = strategy_agents.strategy1_MACD(data)
    # strategy1_macd_signal = strategy1_macd_signals[-1]
    # strategy1_macd_explanation = strategy1_macd_explanations[-1]
    # strategy_text =  f"Strategy: MACD\n" + \
    #                  f"Decision: {strategy1_macd_signal}\n" + \
    #                  f"Reasoning: {strategy1_macd_explanation}\n"

    res_params.update({
        "guidance": guidance_text,
        "sentiment": sentiment_text,
        "strategy1": strategy_text_list[0],
        "strategy2": strategy_text_list[1],
        "strategy3": strategy_text_list[2],
        "strategy4": strategy_text_list[3],
        "trading_record1": trading_record_text_list[0],
        "trading_record2": trading_record_text_list[1],
        "trading_record3": trading_record_text_list[2],
        "trading_record4": trading_record_text_list[3],
        "buy_and_hold_record": buy_and_hold_record
    })

    return res_params

def prepare_latest_market_intelligence_params(state: Dict,
                                            info: Dict,
                                            params: Dict,
                                            memory: MemoryInterface = None,
                                            provider: EmbeddingProvider = None,
                                            diverse_query: DiverseQuery = None
                                            ):

    res_params = deepcopy(params)

    latest_market_intelligence_query = params["latest_market_intelligence_query"]

    query_res = {}
    for query_type, quey_text in latest_market_intelligence_query.items():

        if len(quey_text) == 0 or len(quey_text.split(" ")) <= 5:
            continue

        query_params = {
            "type": "market_intelligence",
            "symbol": params["asset_symbol"],
            "query_text": quey_text,
        }

        query_type = extract_query_type(query_type)
        query_items = diverse_query.query(params=query_params,
                                          query_types=[query_type],
                                          top_k=3)[query_type]["query_items"]

        for item in query_items:
            id = item["id"]
            if id not in query_res:
                query_res[id] = item

    query_res = sorted(query_res.items(), key=lambda x: x[0], reverse=False)
    query_res = [item[1] for item in query_res]

    print(f"Number of queried past market intelligence: {len(query_res)}")

    past_market_intelligence_list = []

    for item in query_res:
        date = item["date"] if isinstance(item["date"], str) else item["date"].strftime("%Y-%m-%d")
        id = item["id"]
        title = item["title"]
        text = item["text"]
        open = item["open"]
        high = item["high"]
        low = item["low"]
        close = item["close"]
        adj_close = item["adj_close"]

        past_market_intelligence_query_item = f"Date: {date}.\n"

        past_market_intelligence_query_item += f"ID: {id}\n" + \
                                               f"Headline: {title}\n" + \
                                               f"Content: {text}\n"
        if math.isnan(open) == False:
            past_market_intelligence_query_item += f"Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), Adj Close: ({adj_close})\n"
        else:
            past_market_intelligence_query_item += f"Prices: Today is closed for trading.\n"

        past_market_intelligence_list.append(past_market_intelligence_query_item)

    if len(past_market_intelligence_list) == 0:
        past_market_intelligence_text = "There is no past market_intelligence.\n"
    else:
        past_market_intelligence_text = "\n".join(past_market_intelligence_list)

    res_params.update({
        "past_market_intelligence": past_market_intelligence_text,
    })

    return res_params

def prepare_low_level_reflection_params(state: Dict,
                                        info: Dict,
                                        params: Dict,
                                        memory: MemoryInterface = None,
                                        provider: EmbeddingProvider = None,
                                        diverse_query: DiverseQuery = None
                                        ):
    res_params = deepcopy(params)

    low_level_reflection_query = params["low_level_reflection_query"]
    low_level_reflection_reasoning = params["low_level_reflection_reasoning"]

    low_level_reflection_short_term_reasoning = low_level_reflection_reasoning["short_term_reasoning"]
    low_level_reflection_medium_term_reasoning = low_level_reflection_reasoning["medium_term_reasoning"]
    low_level_reflection_long_term_reasoning = low_level_reflection_reasoning["long_term_reasoning"]

    date = info["date"]

    latest_low_level_reflection = f"Date: {date}\nShort-Term reasoning: {low_level_reflection_short_term_reasoning}.\nMedium-Term reasoning: {low_level_reflection_medium_term_reasoning}.\nLong-Term reasoning: {low_level_reflection_long_term_reasoning}.\n"

    query_text = low_level_reflection_query

    query_params = {
        "type": "low_level_reflection",
        "symbol": params["asset_symbol"],
        "query_text": query_text,
    }

    query_res = diverse_query.query(params=query_params, query_types=["plain"])

    past_low_level_reflection_list = []

    for query_type, values in query_res.items():
        query_items = values["query_items"]

        type_list = []

        for item in query_items:

            reasoning = item["reasoning"]

            past_low_level_short_term_reasoning_item = reasoning["short_term_reasoning"]
            past_low_level_medium_term_reasoning_item = reasoning["medium_term_reasoning"]
            past_low_level_long_term_reasoning_item = reasoning["long_term_reasoning"]

            type_text = f"Date: {item['date']}\nShort-Term reasoning: {past_low_level_short_term_reasoning_item}\nMedium-Term reasoning: {past_low_level_medium_term_reasoning_item}\nLong-Term reasoning: {past_low_level_long_term_reasoning_item}\n"
            type_list.append(type_text)

        if len(type_list) != 0:
            type_text = "\n\n".join(type_list)
            type_text = "The past low level reflection for " + query_type + " is:\n" + type_text
            past_low_level_reflection_list.append(type_text)

    if len(past_low_level_reflection_list) != 0:
        past_low_level_reflection = "\n\n".join(past_low_level_reflection_list)
    else:
        past_low_level_reflection = "There is no past low level reflection as it is trading initialised."

    res_params.update({
        "latest_low_level_reflection": latest_low_level_reflection,
        "past_low_level_reflection": past_low_level_reflection,
    })

    return res_params

def prepare_high_level_reflection_params(state: Dict,
                                         info: Dict,
                                         params: Dict,
                                         memory: MemoryInterface = None,
                                         provider: EmbeddingProvider = None,
                                         diverse_query: DiverseQuery = None
                                         ):
    res_params = deepcopy(params)

    date = info["date"]

    latest_high_level_reasoning = params["high_level_reasoning"]
    latest_high_level_improvement = params["high_level_improvement"]
    latest_high_level_summary = params["high_level_summary"]
    latest_high_level_query = params["high_level_query"]

    latest_high_level_reflection = f"Date: {date}\nReasoning: {latest_high_level_reasoning}\nImprovement: {latest_high_level_improvement}\nSummary: {latest_high_level_summary}\n"

    query_text = latest_high_level_query

    query_params = {
        "type": "high_level_reflection",
        "symbol": params["asset_symbol"],
        "query_text": query_text,
    }

    query_res = diverse_query.query(params=query_params, query_types=["plain"])

    past_high_level_reflection_list = []

    for query_type, values in query_res.items():
        query_items = values["query_items"]

        type_list = []

        for item in query_items:
            past_high_level_reasoning_item = item["reasoning"]
            past_high_level_improvement_item = item["improvement"]
            past_high_level_summary_item = item["summary"]

            type_text = f"Date: {item['date']}\nReasoning: {past_high_level_reasoning_item}\nImprovement: {past_high_level_improvement_item}\nSummary: {past_high_level_summary_item}\n"
            type_list.append(type_text)

        if len(type_list) != 0:
            type_text = "\n\n".join(type_list)
            type_text = "The past high level reflection for " + query_type + " is:\n" + type_text
            past_high_level_reflection_list.append(type_text)

    if len(past_high_level_reflection_list) != 0:
        past_high_level_reflection = "\n\n".join(past_high_level_reflection_list)
    else:
        past_high_level_reflection = "There is no past high level reflection as it is trading initialised."

    res_params.update({
        "latest_high_level_reflection": latest_high_level_reflection,
        "past_high_level_reflection": past_high_level_reflection,
    })

    return res_params
