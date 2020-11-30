from google.cloud import bigquery

"""
  PRICE DATA FEATURES START
"""
import datetime as dt
import pandas_datareader.data as web
import numpy as np
import math
import csv
import time
start_time = time.time()

class Stock(object):
  def __init__(self, ticker, name="Stock"):
    self.name = name
    self.ticker = ticker
    self.prices = None

  def __str__(self):
    return self.name, self.ticker

  def gen_prices(self, start, end):
    df = web.DataReader(self.ticker, "yahoo", start, end)
    self.prices = df

def trim(stock, market):
  stockarray = np.array([])
  marketarray = np.array([])
  date_list = [end - dt.timedelta(days=x) for x in range(time_period)]
  date_list = list(reversed(date_list))
  for date in date_list:
    try:
      if market.prices.ix[date.strftime("%Y-%m-%d")].ix["Close"]:
        if stock.prices.ix[date.strftime("%Y-%m-%d")].ix["Close"]:
          marketarray = np.append(marketarray,(market.prices.ix[date.strftime("%Y-%m-%d")].ix["Close"],) )
          stockarray = np.append(stockarray, (stock.prices.ix[date.strftime("%Y-%m-%d")].ix["Close"],) )
    except KeyError:
        pass
  return (stockarray, marketarray)

def standardize(stock):
  return list(map(lambda i: 0 if stock[i + 1] == 0 else (stock[i + 1] - stock[i]) / (stock[i]) * 100, range(len(stock) - 1)))

def covar(A, B):
  if len(A) == 0:
    return 0
  else:
    return np.cov(A, B)[0][1] * (len(A) - 1) / len(A)

def var(A):
  return np.var(A)

def beta(Stock, Market):
  Stock = standardize(Stock)
  Market = standardize(Market)
  value = covar(Stock, Market) / var(Market)
  return value

def seperate(Market, Stock):
  BearMarket = list()
  BullMarket = list()
  BearStock = list()
  BullStock = list()
  Market = standardize(Market)
  Stock = standardize(Stock)
  for i in range(len(Market)):
      if Market[i] >= 0:
          BullMarket.append(Market[i])
          BullStock.append(Stock[i])
      else:
          BearMarket.append(Market[i])
          BearStock.append(Stock[i])
  return [BearMarket, BullMarket, BearStock, BullStock]

def stratBetaCalc(BearMarket, BullMarket, BearStock, BullStock):
  BetaBear = covar(BearMarket, BearStock) / var(BearMarket)
  BetaBull = covar(BullMarket, BullStock) / var(BullMarket)
  return [BetaBull, BetaBear]

def stratBeta(Stock, Market):
  BetaList = seperate(Market, Stock)
  return stratBetaCalc(BetaList[0], BetaList[1], BetaList[2], BetaList[3])

def lineFinder(Point1, Point2):
  slope = (Point2[1] - Point1[1]) / (Point2[0] - Point1[0])
  intercept = Point1[1] - slope * Point1[0]
  return [slope, intercept]

def residualArray(xaxis, Line, Pricelist):
  residualArray = []
  for i in range(0, len(Pricelist)):
      residualArray.append(Pricelist[i] - (Line[0] * xaxis[i] + Line[1]))
  return residualArray

def volatilityFinder(PriceList):
  xaxis = []
  for i in range(0, len(PriceList)):
      xaxis.append(i)
  Line = lineFinder([xaxis[0], PriceList[0]], [xaxis[len(xaxis) - 1], PriceList[len(PriceList) - 1]])
  ResidArray = residualArray(xaxis, Line, PriceList)
  return math.sqrt(var(ResidArray))

def directionFinder(Pricelist):
  if Pricelist[len(Pricelist) - 1] < Pricelist[0]:
      if Pricelist[len(Pricelist) - 1] == 0:
          return 0
      else:
          return str(
              round(100 * (Pricelist[0] - Pricelist[len(Pricelist) - 1]) / (Pricelist[len(Pricelist) - 1]), 2)) + '%'
  elif Pricelist[len(Pricelist) - 1] > Pricelist[0]:
      if Pricelist[len(Pricelist) - 1] == 0:
          return 0
      else:
          str(round(100 * (Pricelist[0] - Pricelist[len(Pricelist) - 1]) / (Pricelist[len(Pricelist) - 1]), 2)) + '%'
  else:
      return 0

def StockStat(StockName, Stock, Market):
  Stock, Market = trim(Stock, Market)
  StockS = standardize(Stock)
  MarketS = standardize(Market)
  StratList = stratBeta(Stock, Market)
  BetaDirList = stratBeta(Stock, Market)
  ReturnsMeanData = seperate(Market, Stock)
  StockInfo = {
    'Stock': StockName,
    #'Start Price': Stock[0],
    #'Finish Price': Stock[-1],
    #'Period High':max(Stock) ,
    #'Period Low': min(Stock),
    'Gain/Loss %': str((Stock[-1] - Stock[0]) * 100 / (Stock[0])),
    'Beta': beta(Stock, Market),
    'Beta-Bull': BetaDirList[0],
    'Beta-Bear':BetaDirList[1],
    'Market Correlation': covar(StockS, MarketS) / (math.sqrt(var(StockS) * var(MarketS))),
    'Average Daily Return': sum(StockS) / len(StockS),
    'Returns Bull': sum(ReturnsMeanData[3]) / len(ReturnsMeanData[3]),
    'Returns Bear': sum(ReturnsMeanData[2]) / len(ReturnsMeanData[2])
  }

  return list(StockInfo.values()) #Dicts maintian correct order in python 3.6 lol very bad but i'm lazy

"""
  PRICE DATA FEATURES END
"""

"""
Creates price features.
"""

# Fetch raw price series from Bigquery.
QUERY = """
  SELECT
    *
  FROM
    `silicon-badge-274423.features.sp_daily_features`
"""

client = bigquery.Client(project='silicon-badge-274423')
df = client.query(QUERY).to_dataframe()
df = df.dropna().sort_values(by=['permno', 'date'])

import pdb; pdb.set_trace()

# Create 1 year windows for all permnos.

# Translate into returns.

# Get fundamental data.

# Z-score transform these windows and add to dataframe.

# Upload to Bigquery.
