from distutils.log import error
from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--df_path",
    type=str,
    default="data/data/OE_BTC/test.csv",
    help="the path for the downloaded data to generate the environment")
parser.add_argument("--initial_amount",
                    type=int,
                    default=100000,
                    help="the initial amount of money for trading")
parser.add_argument("--tech_indicator_list",
                    type=list,
                    default=['midpoint','spread','buys','sells','bids_distance_0','bids_distance_1','bids_distance_2','bids_distance_3','bids_distance_4',
                    'bids_distance_5','bids_distance_6','bids_distance_7','bids_distance_8','bids_distance_9','bids_distance_10','bids_distance_11',
                    'bids_distance_12','bids_distance_13','bids_distance_14','bids_notional_0','bids_notional_1','bids_notional_2','bids_notional_3',
                    'bids_notional_4','bids_notional_5','bids_notional_6','bids_notional_7','bids_notional_8','bids_notional_9','bids_notional_10',
                    'bids_notional_11','bids_notional_12','bids_notional_13','bids_notional_14','bids_cancel_notional_0','bids_cancel_notional_1',
                    'bids_cancel_notional_2','bids_cancel_notional_3','bids_cancel_notional_4','bids_cancel_notional_5','bids_cancel_notional_6',
                    'bids_cancel_notional_7','bids_cancel_notional_8','bids_cancel_notional_9','bids_cancel_notional_10','bids_cancel_notional_11',
                    'bids_cancel_notional_12','bids_cancel_notional_13','bids_cancel_notional_14','bids_limit_notional_0','bids_limit_notional_1',
                    'bids_limit_notional_2','bids_limit_notional_3','bids_limit_notional_4','bids_limit_notional_5','bids_limit_notional_6',
                    'bids_limit_notional_7','bids_limit_notional_8','bids_limit_notional_9','bids_limit_notional_10','bids_limit_notional_11',
                    'bids_limit_notional_12','bids_limit_notional_13','bids_limit_notional_14','bids_market_notional_0','bids_market_notional_1',
                    'bids_market_notional_2','bids_market_notional_3','bids_market_notional_4','bids_market_notional_5','bids_market_notional_6',
                    'bids_market_notional_7','bids_market_notional_8','bids_market_notional_9','bids_market_notional_10','bids_market_notional_11',
                    'bids_market_notional_12','bids_market_notional_13','bids_market_notional_14','asks_distance_0','asks_distance_1','asks_distance_2',
                    'asks_distance_3','asks_distance_4','asks_distance_5','asks_distance_6','asks_distance_7','asks_distance_8','asks_distance_9',
                    'asks_distance_10','asks_distance_11','asks_distance_12','asks_distance_13','asks_distance_14','asks_notional_0','asks_notional_1',
                    'asks_notional_2','asks_notional_3','asks_notional_4','asks_notional_5','asks_notional_6','asks_notional_7','asks_notional_8',
                    'asks_notional_9','asks_notional_10','asks_notional_11','asks_notional_12','asks_notional_13','asks_notional_14','asks_cancel_notional_0',
                    'asks_cancel_notional_1','asks_cancel_notional_2','asks_cancel_notional_3','asks_cancel_notional_4','asks_cancel_notional_5',
                    'asks_cancel_notional_6','asks_cancel_notional_7','asks_cancel_notional_8','asks_cancel_notional_9','asks_cancel_notional_10',
                    'asks_cancel_notional_11','asks_cancel_notional_12','asks_cancel_notional_13','asks_cancel_notional_14','asks_limit_notional_0',
                    'asks_limit_notional_1','asks_limit_notional_2','asks_limit_notional_3','asks_limit_notional_4','asks_limit_notional_5',
                    'asks_limit_notional_6','asks_limit_notional_7','asks_limit_notional_8','asks_limit_notional_9','asks_limit_notional_10',
                    'asks_limit_notional_11','asks_limit_notional_12','asks_limit_notional_13','asks_limit_notional_14','asks_market_notional_0',
                    'asks_market_notional_1','asks_market_notional_2','asks_market_notional_3','asks_market_notional_4','asks_market_notional_5',
                    'asks_market_notional_6','asks_market_notional_7','asks_market_notional_8','asks_market_notional_9','asks_market_notional_10',
                    'asks_market_notional_11','asks_market_notional_12','asks_market_notional_13','asks_market_notional_14'],
                    help="the features we want to presents in the environment")
parser.add_argument("--length_keeping",
                    type=int,
                    default=30,
                    help="the number of timestamp we want to keep")
class TradingEnv(gym.Env):
    #here is an example of data-driven OE environment
    # the state is the corresponding order book in that time frame
    # the action is the volume of we want to sell or buy and the price we set
    def __init__(self, config):
        self.time_frame = 0
        self.df = pd.read_csv(config["df_path"], index_col=0)
        self.initial_amount = config["initial_amount"]
        self.tech_indicator_list = config["tech_indicator_list"]
        self.portfolio=[self.initial_amount]+[0]+[0]+[0]
        self.portfolio_value_history=[self.initial_amount]
        self.portfolio_history=[self.portfolio]
        # ?????????order history,?????????????????????????????????????????? ???????????????check ??????????????????order???????????????????????? ?????????????????? ????????????????????????????????????order
        # order history??????????????????????????? 1. place order????????????volume???????????????????????????????????? ?????????????????????level?????????????????? ???????????? 
        # ?????? ??????midpoint???????????????level ??????level?????????????????? ??????????????????????????? ??????0?????? ???????????? ?????????notional?????? ???????????? ???????????????????????????level????????????cancel volume
        # ??????cancel volume??????????????????????????????volume??????cancel volume?????? ??????0?????????2. ????????????????????? +?????? -?????? ??????????????????short ?????????portfolio???????????? 
        # ?????????????????? ?????? ??????????????????????????????????????? ???????????? ?????????????????????????????????  ??????????????????????????? ???3. ??????????????? ??????????????????????????????????????????order???????????????????????????level
        self.order_length=config["length_keeping"]
        self.order_history=[]
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,))
        #set action????????????volume ????????? ????????? ?????????????????? 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list),))
        self.data = self.df.loc[self.time_frame, :]
        self.state = np.array(self.data[self.tech_indicator_list].values.tolist())
        self.terminal = False
        self.rewards=0

    def reset(self):
        self.time_frame = 0
        self.portfolio=[self.initial_amount]+[0]+[0]+[0]
        self.portfolio_history=[self.portfolio]
        self.order_history=[]
        self.data = self.df.loc[self.time_frame, :]
        self.state = np.array(self.data[self.tech_indicator_list].values.tolist())
        self.terminal = False
        self.rewards=0
        return self.state

    def step(self,action:np.array):
        # for the step ones we first check the dimension, then we check whether it is terminal
        # if it is not terminal, we first check whether you want to buy or sell 
        # if you want to buy, we first use the single price times the amount and compare it with our cash, if you do not have enough cash, we will
        # use the all the cash divded by the single price you ask and make an order like this(TODO: this is a little bizzare because we will check the order again
        # when we have a chance to trade and it is largely depend on the cash back then if your cash is not enough we just place the order we can
        # afford and cancell the rest )
        # if you want to sell, we first have to check the whether you are currently holding and if you do not have enough bitcoin, then we will shrink it to the share 
        # you have at this moment and hold the order, we will double check it at the moment when trades could happen and if we do not have enough share
        # we will cancell the rest order as well 

        # check the dimension
        if action.shape!=(2,):
            raiseExceptions("sorry, the dimension of action is not correct, the dimension should be (2,), where the first dimension is the \
                volume you want to trade and the second dimension is the price you wanna sell or buy")
        # check the amount of money you have or bitcoin is enough for the current order
        self.portfolio=self.portfolio_history[-1]
        # portfolio ???4????????? 1. free cash 2. cash will be used for existing order in the LOB 3.free bitcoin 4. bitcoin in order that is listed in the order book for sell
        # notice 1+2 the amount of cash we are currntely holding and 3+4 is the bitcoin we are holding
        # according to the order 1 and 3 we modify our action and make the order tradable 
        #TODO we need to modify the definition of portfolio and the very end as well and add it into the portfolio_history part
        # the think is that we check the order history and clarify the cash and bitcoin taken in the order book and 
        if action[0]<0:
            # if the action's volume is smaller than 0, we are going to sell the bitcoin we are holding
            sell_volume=-action[0]
            if self.portfolio[2]<=sell_volume:
                sell_volume=self.portfolio[2]
            action=[-sell_volume,action[1]]
            if action[1]<self.data["midpoint"]*(1+self.data["asks_distance_0"]*0.01) or action[1]>self.data["midpoint"]*(1+self.data["asks_distance_14"]*0.01):
                start_volume=0
            else:
                start_volume=0
                for i in range(15):
                    if action[1]==self.data["midpoint"]*(1+self.data["asks_distance_{}".format(i)]*0.01
                    ):
                        start_volume=self.data["asks_notional_{}".format(i)]
            action=[-sell_volume,action[1],start_volume]

        if action[0]>0:
            # if the action's volume is greater than 0, we are going to buy the bitcoin we are holding
            buy_volume=action[0]
            buy_money=buy_volume*action[1]
            if buy_money>self.portfolio[0]:
                buy_volume=self.portfolio[0]/action[1]
            action=[buy_volume,action[1]]
            if action[1]>self.data["midpoint"]*(1+self.data["bids_distance_0"]*0.01) or action[1]<self.data["midpoint"]*(1+self.data["bids_distance_14"]*0.01):
                start_volume=0
            else:
                start_volume=0
                for i in range(15):
                    if action[1]==self.data["midpoint"]*(1+self.data["bids_distance_{}".format(i)]):
                        start_volume=self.data["bids_notional_{}".format(i)]
            action=[buy_volume,action[1],start_volume]
        if action[0]==0:
            action=[0,0,0]
        order=action
        self.order_history.append(order)
        if len(self.order_history)>self.order_length:
            self.order_history.pop(0)
        #????????????order_history???????????????????????????????????? ??????????????????????????????????????????????????????
        #??????????????????????????? ???????????????????????????????????????????????????order ??????????????????
        #????????????????????? ????????????step????????????????????? ????????????start volume???????????????start volume??????0???order????????? ????????????????????????????????????level ??????????????????
        #???0 ?????????????????????cancel_order???????????? ??????start_volume???0?????? 
        previous_data=self.data
        self.time_frame=self.time_frame+1
        self.data=self.df.loc[self.time_frame, :]
        for i in range(len(self.order_history)):
            order=self.order_history[i]
            start_volume=order[2]
            if order[0]<0:
                # ?????? ???ask
                if order[2]!=0:
                    if order[1]<self.data["midpoint"]*(1+self.data["asks_distance_0"]*0.01) or order[1]>self.data["midpoint"]*(1+self.data["asks_distance_14"]*0.01):
                        order[2]=0
                    else:
                        order[2]=0
                        for i in range(15):
                            if order[1]==self.data["midpoint"]*(1+self.data["asks_distance_{}".format(i)]*0.01):
                                order[2]=max(0,start_volume-self.data["asks_cancel_notional_{}".format(i)])
            if order[0]>0:
                if order[2]!=0:
                    if order[1]>self.data["midpoint"]*(1+self.data["bids_distance_0"]*0.01) or order[1]<self.data["midpoint"]*(1+self.data["bids_distance_14"]*0.01):
                        order[2]=0
                    else:
                        order[2]=0
                        for i in range(15):
                            if order[1]==self.data["midpoint"]*(1+self.data["bids_distance_{}".format(i)]*0.01):
                                order[2]=max(0,start_volume-self.data["bids_cancel_notional_{}".format(i)])
            self.order_history[i]=order
        # ?????????start_volume???????????????????????????order???????????????????????? ??????????????????self.portfolio???self.portfolio_history
        # ?????????self.portfolio??????????????????????????????place an order ??????????????????orderbook????????????????????????????????? ??????????????????????????????
        #????????????????????????self.portfolio ?????????order??????????????????????????? ???????????????self.portfolio????????? ??????????????? 
        all_cash=self.portfolio[0]+self.portfolio[1]
        all_bitcoin=self.portfolio[2]+self.portfolio[3]
        old_portfolio_value=self.portfolio[0]+self.portfolio[1]+previous_data["midpoint"]*(self.portfolio[2]+self.portfolio[3])
        self.portfolio_value_history.append(old_portfolio_value)
        ordered_cash=0
        ordered_bitcoin=0
        for i in range(len(self.order_history)):
            if self.order_history[i][0]<0:
                ordered_bitcoin=ordered_bitcoin-self.order_history[i][0]
            if self.order_history[i][0]>0:
                ordered_cash=ordered_cash+self.order_history[i][0]*self.order_history[i][1]
        free_cash=all_cash-ordered_cash
        free_bitcoin=all_bitcoin-ordered_bitcoin
        self.portfolio=[free_cash,ordered_cash,free_bitcoin,ordered_bitcoin]
        if free_cash<0 or free_bitcoin<0:
            raise error("Something is wrong witht the order you place and there is no enough free cash or bitcoin in our portfolios, \
            the current portfolio is {}".format(self.portfolio))
            #order execution ???portfolio????????? ???ordered_cash??????free bitcoin ?????????ordered bitcoin ??????free cash
        for i in range(len(self.order_history)):
            order=self.order_history[i]
            if order[0]<0:
                #???????????????????????????bid?????????distance????????? ????????????????????????volume?????? ????????? ?????????????????????
                if order[1]<self.data["midpoint"]*(1+self.data["bids_distance_0"]*0.01):
                    # the order could be executed now let us see what is the greatest bargin
                    # ???????????? ??????????????????????????????
                    
                    self.portfolio=[self.portfolio[0]-order[0]*order[1],self.portfolio[1],self.portfolio[2],self.portfolio[3]+order[0]]
                    order=[0,0,0]
                elif order[1]==self.data["midpoint"]*(1+self.data["bids_distance_0"]*0.01):
                    #???volume??? 
                    tradable_volume=min(self.data["bids_notional_0"]-order[2],-order[0])
                    if tradable_volume==-order[0]:
                        self.portfolio=[self.portfolio[0]-order[0]*order[1],self.portfolio[1],self.portfolio[2],self.portfolio[3]+order[0]]
                        order=[0,0,0]
                    else:
                        order[0]=order[0]+tradable_volume
                        order[2]=0
                        self.portfolio=[self.portfolio[0]+tradable_volume*order[1],self.portfolio[1],self.portfolio[2],self.portfolio[3]-tradable_volume]
            if order[0]>0:
                if order[1]>self.data["midpoint"]*(1+self.data["asks_distance_0"]*0.01):
                        self.portfolio=[self.portfolio[0],self.portfolio[1]-order[0]*order[1],self.portfolio[2]+order[0],self.portfolio[3]]
                        order=[0,0,0]
                elif order[1]==self.data["midpoint"]*(1+self.data["asks_distance_0"]*0.01):
                    tradable_volume=min(self.data["asks_notional_0"]-order[2],order[0])
                    if tradable_volume==order[0]:
                        self.portfolio=[self.portfolio[0],self.portfolio[1]-order[0]*order[1],self.portfolio[2]+order[0],self.portfolio[3]]
                        order=[0,0,0]
                    else:
                        order[0]=order[0]-tradable_volume
                        order[2]=0
                        self.portfolio=[self.portfolio[0],self.portfolio[1]-tradable_volume*order[1],self.portfolio[2]+tradable_volume,self.portfolio[3]]
            self.order_history[i]=order
        self.portfolio_history.append(self.portfolio)
        new_portfolio_value=self.portfolio[0]+self.portfolio[1]+self.data["midpoint"]*(self.portfolio[2]+self.portfolio[3])
        self.portfolio_value_history.append(new_portfolio_value)
        self.reward=self.portfolio_value_history[-1]-self.portfolio_value_history[-2]
        self.rewards=self.rewards+self.reward
        self.state=np.array(self.data[self.tech_indicator_list].values.tolist())
        self.terminal = (self.time_frame+1>=self.df.index[-1])
        if self.terminal:
            print("done!")
            print("the accmulated rewards is{}".format(self.rewards))
        return self.state, self.reward, self.terminal, {}
if __name__=="__main__":
    args = parser.parse_args()
    a = TradingEnv(vars(args))
    state = a.reset()
    print(state.shape)
    action=np.array([1,56000])
    done=False
    while not done:
        state,reward,done,_=a.step(action)
        print(a.portfolio)

    print(a.portfolio)
    print(a.portfolio_value_history[-1])

