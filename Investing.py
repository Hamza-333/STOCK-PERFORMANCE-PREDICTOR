import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, svm, preprocessing
from datetime import datetime
import yahoo_fin.stock_info as yf
import numpy as np
from collections import Counter
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


def create_df(d):
    with open('sp500.csv', 'r') as f:
        sp500 = f.read()
    sp500 = (sp500.split(','))

    for comp in sp500[:25]:
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
    val = df.values
    new_df = pd.DataFrame()
    for num in df.columns:
        df[num] = df[num].fillna(0)
        for val in range(len(df[num])):

            if isinstance(df[num][val], str):
                if ',' in df[num][val]:
                    pass
                elif 'M' in df[num][val]:
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
                # elif 'NaN' == df[num][val] or 'nan' == df[num][val]:
                #     df[num][val] = 0
                elif '%' in df[num][val]:
                    df[num][val]  = float(df[num][val][:-1])
                elif df[num][val].isnumeric():
                    df[num][val] = float(df[num][val])   
            else:
                continue
    df.to_csv('df_cleaned.csv')
def historic_data():
    with open('sp500.csv', 'r') as f:
        sp500 = f.read()
    sp500 = (sp500.split(','))
    df = yf.get_data(sp500[0], start_date = '01/04/2020', end_date = '01/04/2021', interval = '1mo')
    for i in range(1, 25):
        d = yf.get_data(sp500[i], start_date = '01/04/2020', end_date = '01/04/2021', interval = '1mo')
        df = df.append(d, ignore_index = False)
    sp500_df = yf.get_data('^GSPC', start_date = '01/04/2020', end_date = '01/04/2021', interval = '1mo')
  
   
    df.to_csv('historic_data.csv')
    sp500_df.to_csv('sp500_data.csv')


def Performance():
    df = pd.read_csv('df_cleaned.csv')
    historic_df = pd.read_csv('historic_data.csv')
    sp500_df = pd.read_csv('sp500_data.csv')

    tmp = {'Performance': []}
    start_date = '2020-02-01'
    end_date = '2021-01-01'
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

            if diff > 5:
                tmp['Performance'].append('Outperform')
            else:
                tmp['Performance'].append('Underperform')
            flag = True
        else:
            flag = False     
    tmp_df = pd.DataFrame(tmp)
    df = pd.concat([df, tmp_df], axis = 1)
    df.to_csv('Performance_df.csv')      

def Dataset():

    df = pd.read_csv('Performance_df.csv')
    df = df.reindex(np.random.permutation(df.index))
    
    lst = ['Beta (5Y Monthly)', '52-Week Change 3',
       '52 Week High 3', '52 Week Low 3', '50-Day Moving Average 3',
       '200-Day Moving Average 3', 'Avg Vol (3 month) 3', 'Avg Vol (10 day) 3',
       'Implied Shares Outstanding 6', 'Float', '% Held by Insiders 1',
       '% Held by Institutions 1', 'Forward Annual Dividend Rate 4',
       'Forward Annual Dividend Yield 4', 'Trailing Annual Dividend Rate 3',
       'Trailing Annual Dividend Yield 3', '5 Year Average Dividend Yield 4',
       'Payout Ratio 4',
        'Profit Margin', 'Operating Margin (ttm)',
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
    Z = df['Ticker'].values
  
    return X, y, Z

def Analysis():
    test_size = 10

    X, y, Z = Dataset()
    
    invest = []
    clf = svm.SVC(kernel = 'linear', C= 1.0)
    clf.fit(X[:-test_size], y[:-test_size])
   
    count = 0

    for i in range(test_size + 1):

        if clf.predict([X[-i]])[0] == y[-i]:
            count += 1
        if clf.predict([X[i]])[0] == 1:
            invest.append(Z[i])

    print('Accuracy:', (count/test_size) * 100)

    return invest
if __name__ == '__main__':
    create_df(d)
    clean_data()
    historic_data()
    Performance
    iterations = 20
    common_lst = []
    for i in range(iterations):
        lst = Analysis()
        for stock in lst:
            common_lst.append(stock)
    final = []
    x = Counter(common_lst)
    for stock in x:
        if x[stock] >= iterations * 0.25:
            final.append(stock)
    print('Companies to invest:', final)

