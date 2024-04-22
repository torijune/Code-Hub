# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

SYMBOLS = ['BrentOil', 'Copper', 'CrudeOil', 'Gasoline', 'Gold', 'NaturalGas', 'Platinum', 'Silver',
           'AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']  # !!! 수정 금지 !!!


class PriceDataset(Dataset):
    def __init__(self,
                 start_date,
                 end_date,
                 is_training=True,
                 in_columns=['USD_Price'],
                 out_columns=['USD_Price'],
                 input_days=3,
                 data_dir='data'):
        self.x, self.y = make_features(start_date, end_date,
                                       in_columns, out_columns, input_days,
                                       is_training,  data_dir)

        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # self.x[idx]의 사이즈는 현재 (input_days, input_dim)이므로, 이를 (input_days * input_dim)으로 flatten함
        return torch.flatten(self.x[idx]), self.y[idx]


def make_features(start_date, end_date, in_columns, out_columns, input_days,
                  is_training, data_dir='data'):

    start, end = ''.join(start_date.split('-'))[2:], ''.join(end_date.split('-'))[2:]
    save_fname = f'all_{start}_{end}.pkl'

    if os.path.exists(os.path.join(data_dir, save_fname)):
        print(f'loading from {os.path.join(data_dir, save_fname)}')
        table = pd.read_pickle(os.path.join(data_dir, save_fname))

    else:
        print(f'making features from {start_date} to {end_date}')
        table = merge_data(start_date, end_date, symbols=SYMBOLS, data_dir=data_dir)
        table.to_pickle(os.path.join(data_dir, save_fname))
        print(f'saved to {os.path.join(data_dir, save_fname)}')


    # TODO: 데이터 클렌징 및 전처리
    # 주의 : USD_Price에는 값이 있고, 나머지에는 값이 없는 경우가 있음. 이러한 경우는 삭제되지 않도록 주의할 것
    #       만일 삭제될 경우 test.py에서 에러가 발생하여 0점 처리됨
    table.dropna(inplace=True, subset=['USD_Price'])

    column_means = table.mean()
    table.fillna(column_means,inplace=True)

    # 주의 : 미국 환율 가격을 예측해야 하므로, config.yaml의 out_columns에는 반드시 'USD_Price'가 포함되어야 함
    if 'USD_Price' not in out_columns:
        raise ValueError('USD_Price must be included in out_columns')   # !!! 수정 금지 !!!

    use_columns = list(set(in_columns + out_columns))  # 중복 제거
    df = table[use_columns]

    # TODO: 추가적인 feature engineering이 필요하다면 아래에 작성
    # 가령, 주식 데이터의 경우 이동평균선, MACD, RSI 등의 feature를 생성할 수 있음
    # 주의 : 미래 데이터를 활용하는 일이 없도록 유의할 것 (가령, 10월 31일 데이터(row)에 10월 31일 뒤의 데이터가 활용되면 안 됨)
    # 주의 : 추가로 활용할 feature들은 in_columns에도 추가할 것

    ## MA(Moving_Avg) 추가
    moving_avg_window = 7
    df['MA'] = df['USD_Price'].rolling(window=moving_avg_window, min_periods=1).mean()

    ## 장기(느린) 이동평균 계산
    long_window = 30
    df['Long_MA'] = df['USD_Price'].rolling(window=long_window, min_periods=1).mean()

    ## MACD 계산
    df['MACD'] = df['MA'] - df['Long_MA']

    ## MACD 신호선(7일 기간의 이동평균) 계산
    signal_window = 7
    df['Signal'] = df['MACD'].rolling(window=signal_window, min_periods=1).mean()

    ##EWMA
    df['Smoothed'] = df['USD_Price'].ewm(span=12, adjust=False).mean()

    ## 계절성 분해 고고
    # result = seasonal_decompose(df['USD_Price'], model='multiplicative', period=12)
    # trend = result.trend
    # seasonal = result.seasonal
    # residual = result.resid
    # df['trend'] = trend
    # df['seasonal'] = seasonal
    # df['residual'] = residual

    #df.dropna(inplace=True, subset = ['USD_Price'])

    in_columns += ['MA','Signal','MACD', 'Long_MA','Smoothed']

    # input_days 만큼의 과거 데이터를 사용하여 다음날의 USD_Price를 예측하도록 데이터셋 구성됨
    date_indices = sorted(table.index)
    x = np.asarray([df.loc[date_indices[i:i + input_days], in_columns] for i in range(len(df) - input_days)])
    y = np.asarray([df.loc[date_indices[i + input_days], out_columns] for i in range(len(df) - input_days)])


    # 최근 10일을 test set으로 사용
    # 주의 : 검증 및 테스트 과정에 반드시 최근 10일 데이터를 사용해야 하므로 수정하지 말 것
    training_x, test_x = x[:-10], x[-10:]  # !!! 수정 금지 !!!
    training_y, test_y = y[:-10], y[-10:]  # !!! 수정 금지 !!!


    return (training_x, training_y) if is_training else (test_x, test_y)



def merge_data(start_date, end_date, symbols, data_dir='data'):

    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol, data_dir), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns

        ## 코로나기간 빼기
        # exclude_start = '2020-01-01'
        # exclude_end = '2022-08-29'
        #
        # exclude_period = (df_temp.index >= pd.to_datetime(exclude_start)) & (
        #             df_temp.index <= pd.to_datetime(exclude_end))
        # df_temp = df_temp[~exclude_period]

        df = df.join(df_temp)

    return df




if __name__ == "__main__":

    start_date = '2013-01-01'
    end_date = '2023-10-27'
    is_training = False

    test_data = PriceDataset(start_date, end_date,
                             is_training=is_training,
                             in_columns=['USD_Price', 'Gold_Price', 'Silver_Price'],
                             out_columns=['USD_Price'],
                             input_days=5,
                             data_dir='data')

    print(f'\ntest_data length : {len(test_data)}')
    print(f'\ndataset_x_original[9] : \n{test_data.x[9]}')
    print(f'\ndataset_x_flatten[9] : \n{test_data.__getitem__(9)[0]}')
    print(f'\ndataset_y[9] : \n{test_data.__getitem__(9)[1]}')

    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)
    print(f'\ntest_dataloader length : {len(test_dataloader)}')
