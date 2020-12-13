import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config
import glob
import os

def load_dataset(*, file_path: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
#     _data = pd.read_csv(file_name)

#         file_path = 'ETFs'
#         #'/Users/ziqunye/Documents/stanford/project/Reinforcement-Learning-Project/preprocessing/ETFs'
#         file_path = 

    # abs_path = '/Users/ziqunye/Documents/stanford/project/Reinforcement-Learning-Project/data/'
    abs_path = '/home/scpdxcs/Reinforcement-Learning-Project/ETFs'
    
    print(os.path.join(abs_path, file_path, '*.txt'))
    data_list = []
    STOCK_DIM = 0
    for filename in glob.glob(os.path.join(abs_path, file_path, '*.txt')):
        ticker = os.path.basename(filename).split('.')[0]
        market = os.path.basename(filename).split('.')[1]
        product_type = 'ETFs'#os.path.basename(filename).split('.')[2]
        df = pd.read_csv(filename, ',')
        print(df['Date'].iloc[0])
        
        if int(df['Date'].iloc[0].replace('-','')) <= 20090102:
            # global STOCK_DIM
            STOCK_DIM += 1
            print('STOCK_DIM_preprocess', STOCK_DIM)
            df['ticker'] = ticker 
            df['market'] = market
            df['product_type'] = product_type
            data_list.append(df)

    df_all = pd.concat(data_list, axis=0)
    df_all.columns = ['datadate', 'open', 'high', 'low', 'close', 'volume', 
                  'openint', 'tic', 'market', 'product_type']
    df_all['datadate'] = df_all['datadate'].apply(lambda x: int(x.replace('-', '')))
    print(df_all.shape)
    return df_all, STOCK_DIM

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data


def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    # stock['close'] = stock['close']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def preprocess_data():
    """data preprocessing pipeline"""

    df, STOCK_DIM = load_dataset(file_path='ETF30')
    print('-----------------')
    print(df.shape)
    print('-----------------')
    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = df#calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    print(df)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final, STOCK_DIM

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='close')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










