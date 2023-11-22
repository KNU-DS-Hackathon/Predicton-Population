
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def lstm(df):

    use_cols = ['총인구수(명)', '유치원 수', '초등학교 수', '출생건수', '사망건수', '혼인건수', '이혼건수', '학령인구(명)']
    df['행정구역'].value_counts()

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    # 데이터 전처리
    scaler = MinMaxScaler()
    df[use_cols] = scaler.fit_transform(df[use_cols])


    # 연도가 2013부터 2020까지인 데이터 추출
    filtered_df = df[(df['연도'] >= 2013) & (df['연도'] <= 2020)]

    # 시퀀스 데이터 생성
    def create_sequence_data(data, sequence_length):
        sequences = []
        for region in data['행정구역'].unique().tolist():
            temp_df = data[data['행정구역'] == region]
            if len(temp_df) < sequence_length:
                continue

            temp_df = temp_df[use_cols]
            for i in range(len(temp_df) - sequence_length):
                seq = temp_df[i:i+sequence_length+1]
                sequences.append(seq)
        return np.array(sequences)

    sequence_length = 3  # 시퀀스 길이
    sequences = create_sequence_data(filtered_df, sequence_length)

    sequences.shape

    # 데이터셋 분리
    X = sequences[:, :-1]
    y = sequences[:, -1][:, -1]

    # 훈련/검증 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # early stopping

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # 모델 학습

    model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2, callbacks=[early_stop])


    def invTransform(scaler, data, colName, colNames):
        dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
        dummy[colName] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
        return dummy[colName].values

    predicted_data = model.predict(X_test)

    # 예측 결과 역전환
    #predicted_data = scaler.inverse_transform(predicted_data)
    predicted_data = invTransform(scaler, predicted_data, '학령인구(명)', use_cols).reshape(-1, 1)
    predicted_data = predicted_data.round()
    #y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_true = invTransform(scaler, y_test, '학령인구(명)', use_cols).reshape(-1, 1)


    # 모델 평가
    mse = mean_squared_error(y_true, predicted_data)
    print(f'Mean Squared Error: {mse:.2f}')

    # MAPE 계산 함수
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # MAPE 계산
    mape = calculate_mape(y_true, predicted_data)
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')


    for yt, pr in zip(y_true, predicted_data):
        print(yt, pr)

    prediction_df = df[df['연도'].isin([i for i in range(2018, 2030)])]

    # 시퀀스 데이터 생성
    def create_sequence_data1(data, sequence_length):
        sequences = []
        regions = []
        year = []
        for region in data['행정구역'].unique().tolist():
            temp_df = data[data['행정구역'] == region]
            if len(temp_df) < sequence_length:
                continue

            year.extend([i for i in range(2021, 2031)])
            temp_df = temp_df[use_cols]
            for i in range(len(temp_df) - sequence_length+1):
                regions.append(region)
                seq = temp_df[i:i+sequence_length]
                sequences.append(seq)
        return np.array(sequences), np.array(regions), np.array(year)

    X_sequences, r, y = create_sequence_data1(prediction_df, sequence_length)


    # 예측
    predicted_data = model.predict(X_sequences)
    predicted_data = invTransform(scaler, predicted_data, '학령인구(명)', use_cols).reshape(-1, 1)

    predicted_data.round()

    predicted_df = pd.DataFrame({'연도': y, '행정구역': r, '학령인구(명)': predicted_data.round().flatten()})
    
    return mape




