# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def make_df(df):

    # %%
    data = df

    # %%
    use_cols = ['총인구수(명)', '유치원 수', '초등학교 수', '출생건수', '사망건수', '혼인건수', '이혼건수', '순이동', '학령인구(명)']

    # %%
    df['행정구역'] = df['행정구역(시도)'].astype(str) + ' ' + df['행정구역(시군구)'].astype(str)


    # %%
    for region in df['행정구역'].unique().tolist():
        temp_df = df[df['행정구역'] == region]

    # %%
    drop_list = ['인천광역시 남구',
                '충청북도 청주시',
                '충청북도 청원군',
                '대구광역시 동구',
                '대구광역시 남구',
                '경기도 여주시',
                '인천광역시 미추홀구']

    # %%
    for region in drop_list:
        df = df[df['행정구역'] != region]

    # %%
    df

    # %%
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    from tqdm import tqdm
    
    warnings.filterwarnings(action='ignore')
    
    def forecasting_features(data):
        sido = data['행정구역(시도)'].unique().tolist()[0]
        sigungu = data['행정구역(시군구)'].unique().tolist()[0]
        sido_sigungu = sido + ' ' + sigungu
        year = data['연도'].unique().tolist()[-1]

        future_data = {'행정구역(시도)': np.array([sido for i in range(10)]), '행정구역(시군구)': np.array([sigungu for i in range(10)]), '행정구역': np.array([sido_sigungu for i in range(10)]), '연도': np.array([i for i in range(year+1, year+11)])}

        for col in data.columns:
            if col in ('행정구역(시도)', '행정구역(시군구)', '행정구역', '연도'):
                continue
            else:
                value = data[col]
                # 시계열 데이터로 변환
                time_series = pd.Series(value)

                # ARIMA 모델 생성 및 훈련
                order = (1, 1, 1)  # (p, d, q)
                model = ARIMA(time_series, order=order)
                results = model.fit()

                # 예측
                forecast_steps = 10
                forecast = results.get_forecast(steps=forecast_steps)

                # 예측 결과 및 신뢰 구간 출력
                forecast_mean = forecast.predicted_mean
                confidence_interval = forecast.conf_int()

                if col not in ('총인구수증감률', '학령인구증감률', '지가변동률'):
                    forecast_mean = forecast_mean.round()
                future_data[col] = forecast_mean
        return future_data

    # %%
    for region in tqdm(df['행정구역'].unique().tolist(), desc=region + ': forecasting arima features....' ):
        temp_df = df[df['행정구역'] == region]

        future_data = forecasting_features(temp_df)
        future_df = pd.DataFrame(future_data)
        df = pd.concat([df, future_df])

    # %%
    df.sort_values(by=['행정구역', '연도'], inplace=True)

    
    warnings.filterwarnings(action='default')

    return df
