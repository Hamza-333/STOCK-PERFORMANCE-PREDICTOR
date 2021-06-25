import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, svm, preprocessing
import datapackage
import os, time
from datetime import datetime
import yahoo_fin.stock_info as yf
import html5lib
import numpy as np

data_url = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'

package = datapackage.Package(data_url)


resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
        
data.to_csv('Sector_info.csv')

data = pd.read_csv('msft.csv')
att = data['Attribute'].to_list()
att.insert(0, 'Ticker')
val = data['Value'].to_list()
val.insert(0, 'MSFT')


d = {}

for i in range(len(att)):

    if att[i].startswith('Shares') or att[i].startswith('Short'):
        pass
    else:
        d[att[i]] = []
  
def Narrow_down(ticker, sector):
    
    data = pd.read_csv('Sector_info.csv')
    lst = []
    for val in data.values:
        if val[-1].lower() == sector.lower():
            lst.append(val[1])
    lst.append(ticker)
    return lst

def create_df(ticker, sector, d):

    tickers = Narrow_down(ticker, sector)
    for comp in tickers:
        data = yf.get_stats(comp)
        att = data['Attribute'].to_list()
        att.insert(0, 'Ticker')
        val = data['Value'].to_list()
        val.insert(0, comp)

        for i in range(len(att)):
            try:
                d[att[i]].append(val[i])
            except Exception as e:
                pass
    df = pd.DataFrame(d)
    df.to_csv('Dataframe.csv')
            

def clean_data():
    df = pd.read_csv('Dataframe.csv')
    
    for num in df.columns:
        df[num] = df[num].fillna(0)
        for val in range(len(df[num])):
            if isinstance(df[num][val], str):
                if ',' in df[num][val]:
                    df[num][val] = df[num][val].replace(',', '')
                if 'M' in df[num][val]:
                    try:
                        df[num][val] = float(df[num][val][:-1])
                        df[num][val] *= 1000000
                    except:
                        pass
                elif 'k' in df[num][val]:
                    try:
                        df[num][val] = float(df[num][val][:-1])
                        df[num][val] *= 1000
                    except:
                        pass
                elif 'B' in df[num][val]:
                    try:
                        df[num][val] = float(df[num][val][:-1])
                        df[num][val] *= 1000000000
                    except:
                        pass
                elif 'T' in df[num][val]:
                    try:
                        df[num][val] = float(df[num][val][:-1])
                        df[num][val] *= 1000000000000
                    except:
                        pass
                elif '%' in df[num][val]:
                    try:
                        df[num][val]  = float(df[num][val][:-1])
                    except:
                        pass
                elif df[num][val].isnumeric():
                    try:
                        df[num][val] = float(df[num][val]) 
                    except:
                        pass  
            else:
                continue
    df.to_csv('df_cleaned.csv')
def historic_data(ticker, sector):
    tickers = Narrow_down(ticker, sector)

    df = yf.get_data(tickers[0], start_date = '02/04/2020', end_date = '02/04/2021', interval = '1mo')
    for i in range(1, len(tickers)):
        d = yf.get_data(tickers[i], start_date = '02/04/2020', end_date = '02/04/2021', interval = '1mo')
        df = df.append(d, ignore_index = False)
    sp500_df = yf.get_data('^GSPC', start_date = '02/04/2020', end_date = '02/04/2021', interval = '1mo')
  
   
    df.to_csv('historic_data.csv')
    sp500_df.to_csv('sp500_data.csv')


def Performance():
    df = pd.read_csv('df_cleaned.csv')
    historic_df = pd.read_csv('historic_data.csv')
    sp500_df = pd.read_csv('sp500_data.csv')

    tmp = {'Performance': []}
    start_date = '2020-02-02'
    end_date = '2021-01-02'
    flag = True
    for row in range(len(historic_df.values) - 1):
        ticker = historic_df.values[row][-1]
        if flag:
            start = row            
        if (row + 2 == len(historic_df.values)) or ticker != historic_df.values[row + 1][-1]:
            
            first_price = historic_df.values[start][-3]
            end_price = historic_df.values[row][-3]
            stock_change = round(((end_price - first_price) / first_price) * 100, 2)
            sp500_change = round(((sp500_df.values[-1][-3] - sp500_df.values[0][-3]) / sp500_df.values[0][-3]) * 100, 2)
            diff = stock_change - sp500_change
            if diff > 0:
                tmp['Performance'].append('Outperform')
            else:
                tmp['Performance'].append('Underperform')
            flag = True
        else:
            flag = False     
    tmp_df = pd.DataFrame(tmp)
    df = pd.concat([df, tmp_df], axis = 1)
    df.to_csv('Performance_df.csv')      

    return stock_change, sp500_change

def Dataset():

    df = pd.read_csv('Performance_df.csv')
    df = df.reindex(np.random.permutation(df.index))

    df = df.fillna(0)
    lst = ['Beta (5Y Monthly)', '52-Week Change 3',
       '52 Week High 3', '52 Week Low 3', '50-Day Moving Average 3',
       '200-Day Moving Average 3', 'Avg Vol (3 month) 3', 'Avg Vol (10 day) 3',
       'Implied Shares Outstanding 6', 'Float', '% Held by Insiders 1',
       '% Held by Institutions 1', 'Forward Annual Dividend Rate 4',
       'Forward Annual Dividend Yield 4', 'Trailing Annual Dividend Rate 3',
       'Trailing Annual Dividend Yield 3', '5 Year Average Dividend Yield 4',
       'Payout Ratio 4', 'Profit Margin', 'Operating Margin (ttm)',
       'Return on Assets (ttm)', 'Return on Equity (ttm)', 'Revenue (ttm)',
       'Revenue Per Share (ttm)', 'Quarterly Revenue Growth (yoy)',
       'Gross Profit (ttm)', 'EBITDA', 'Net Income Avi to Common (ttm)',
       'Diluted EPS (ttm)', 'Quarterly Earnings Growth (yoy)',
       'Total Cash (mrq)', 'Total Cash Per Share (mrq)', 'Total Debt (mrq)',
       'Total Debt/Equity (mrq)', 'Current Ratio (mrq)',
       'Book Value Per Share (mrq)', 'Operating Cash Flow (ttm)',
       'Levered Free Cash Flow (ttm)']
   
    X = np.array(df[lst].values)

    y = (df['Performance'].replace('Underperform', 0).replace('Outperform', 1).values)
   
    X = preprocessing.scale(X)

  
    return X, y

def Analysis(stock, sp500, amount, ticker):

    X, y = Dataset()
    invest = []
    clf = svm.SVC(kernel = 'linear', C= 1.0)
    clf.fit(X[:-1], y[:-1])

    count = 0
    if clf.predict([X[-1]]) == 1:
        print('The company will outperform the market by {a}%'.format(a = stock -sp500))
        
    else:
        print('The company will underperform compared to the market')
    algo_return = round(amount + (amount * (stock / 100)), 2)
    market_return = round(amount + (amount* (sp500 / 100)), 2)
    print('Investing in {ticker} will give you a return of {algo} in one year'.format(ticker = ticker, algo = algo_return))
    print('Investing in the S&P500 will give you a return of {sp500} in one year'.format(sp500 = market_return))

if __name__ == '__main__':
    amount = int(input('How much are you looking to invest?'))
    # ticker = 'MSFT'
    sector = 'Information Technology'
    ticker = input('Enter a ticker')
    sector = input('Choose from the following sectors: \n    1. Consumer \n\
    2. Information Technology\n\
    3. Industrials \n\
    4. Utilities\n\
    5. Financials\n\
    6. Materials\n\
    7. Real Estate \n\
    8. Energy\n\
    9. Health Care\n\
    10. Communication Services').lower()
    
    historic_data(ticker, sector)
    stock, sp500 = Performance()
    Analysis(stock, sp500, amount, ticker)

