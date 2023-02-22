# Download Data from Yahoo Finance
In order to build up your own dataset, Yahoo Finance is an open-source platform where you can get access to various types of financial market data such as US stock, forex and cryptocurrency via Yahoo Finance python API(yfinance). 

Here is an example of script downloading Apple's stock data from yfinance, which contains the open, high, low, close, adjusted close price and volume. 

   ```
   import yfinance as yf
   start_date='2009-01-02'
   end_date='2021-01-01'
   df = yf.download('AAPL', start=start_date, end=end_date, interval='1d')
   ```
By modifying the instructions, you can customize your downloaded dataset. 
 

