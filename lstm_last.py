
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

plt.rc('font', family='AppleGothic')

def lstm(path):

    df = pd.read_csv(path)
    use_cols = ['총인구수(명)', '유치원 수', '초등학교 수', '출생건수', '사망건수', '혼인건수', '이혼건수', '학령인구(명)']

    scaler = MinMaxScaler()
    df[use_cols] = scaler.fit_transform(df[use_cols])

    # 연도가 2013부터 2020까지인 데이터 추출
    filtered_df = df[(df['연도'] >= 2010) & (df['연도'] <= 2021)]

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

    # 데이터셋 분리
    X = sequences[:, :-1]
    y = sequences[:, -1][:, -1]

    # 훈련/검증 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def invTransform(scaler, data, colName, colNames):
        dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
        dummy[colName] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
        return dummy[colName].values

    # MAPE 계산 함수
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Specify the number of iterations
    num_iterations = 100
    mape_list = []
    min_mape = float('inf')  # Initialize with positive infinity

    for iteration in range(num_iterations):
        # Create the model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X_train, y_train, epochs=1000, verbose=1)

        # Predict
        predicted_data = model.predict(X_test)
        predicted_data = invTransform(scaler, predicted_data, '학령인구(명)', use_cols).reshape(-1, 1)
        predicted_data = predicted_data.round()
        y_true = invTransform(scaler, y_test, '학령인구(명)', use_cols).reshape(-1, 1)

        # Evaluate MAPE
        mape = calculate_mape(y_true, predicted_data)
        print(f'{iteration}th: Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
        mape_list.append(mape)


        # Check if the current MAPE is the new minimum
        if mape < min_mape:
            min_mape = mape
            # Save the model when a new minimum is reached
            model.save('best_model.h5')

        # Optionally, you can plot the results for each iteration
        # plt.plot(range(len(y_true)), y_true, label='True')
        # plt.plot(range(len(predicted_data)), predicted_data, label='Predicted', linestyle='--')
        # plt.xlabel('Test Case')
        # plt.ylabel('School_Age_Population')
        # plt.legend()
        # plt.show()

    # Calculate and print the average and minimum MAPE
    average_mape = np.mean(mape_list)
    print(f'Average Mean Absolute Percentage Error (MAPE) over {num_iterations} iterations: {average_mape:.2f}%')
    print(f'Minimum Mean Absolute Percentage Error (MAPE) over {num_iterations} iterations: {min_mape:.2f}%')

    prediction_df = df[df['연도'].isin([i for i in range(2019, 2023)])]

    predict_sequences = create_sequence_data(prediction_df, sequence_length)

    # 데이터셋 분리
    predict_X = predict_sequences[:, :-1]
    predict_y = predict_sequences[:, -1][:, -1]


    # Load the best model
    best_model = load_model('best_model.h5')

    predicted_data = best_model.predict(predict_X)

    # 예측 결과 역전환
    #predicted_data = scaler.inverse_transform(predicted_data)
    predicted_data = invTransform(scaler, predicted_data, '학령인구(명)', use_cols).reshape(-1, 1)
    predicted_data = predicted_data.round()
    #y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_true = invTransform(scaler, predict_y, '학령인구(명)', use_cols).reshape(-1, 1)

    # 모델 평가
    mse = mean_squared_error(y_true, predicted_data)
    print(f'Mean Squared Error: {mse:.2f}')

    # MAPE 계산 함수
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # MAPE 계산
    mape = calculate_mape(y_true, predicted_data)
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    return mape, best_model




