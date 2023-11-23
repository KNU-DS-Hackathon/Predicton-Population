# %%
import pandas as pd

def parameterTuning(distance, params):
    filePath = './dataset/adjacency_matrix.csv'
    distance_matrix = pd.read_csv(filePath, index_col=0)


    distance_threshold = distance
    local_params = params
    adj_params = 1 - local_params

    dm_with_threshhold = distance_matrix[distance_matrix < distance_threshold]


    # dm_with_threshhold의 모든 가중치를 역수로 취하기.
    # 가중치가 0일경우 1로 바꾸기.

    dm_with_threshhold = dm_with_threshhold.replace(0, 1)
    dm_with_threshhold = 1 / dm_with_threshhold


    import numpy as np

    # 각 행의 합을 계산
    row_sums = dm_with_threshhold.sum(axis=1) - 1

    # %%
    # 각 인접 도시와의 가중치를 총합으로 나눔
    normalized_df = dm_with_threshhold.div(row_sums, axis=0) * adj_params


    # %%
    # 대각선 행은 0.7으로 만들기

    # 대각선 행 중 inf가 아닌 행은 0.7으로 만들기

    for i in range(len(normalized_df)):
        if normalized_df.iloc[i, i] != np.inf:
            normalized_df.iloc[i, i] = local_params
        else:
            normalized_df.iloc[i, i] = 1


    # %%
    df = pd.read_csv('./dataset/no_NaN_dataset_final.csv')


    # %%
    def find_cityname_in_df(df, cityname, year):
        # cityname에서 첫 두글자는 시도에, 나머지는 시군구에 저장함
        
        sido = cityname[:2]
        sigungu = cityname[2:]
        
        # sido와 sigungu를 모두 포함하는 행 찾기 꼭 일치하지 않아도 되고 일부만 포함해도 됨
        
        sido_df = df[df['행정구역(시도)'].str.contains(sido)]
        # sigungu_df = sido_df[sido_df['행정구역(시군구)'].str.contains(sigungu)]
        sigungu_df = sido_df[sido_df['행정구역(시군구)'] == sigungu]
        
        
        result = sigungu_df[sigungu_df['연도'] == year]
        
        if len(result) == 0:
            sido = cityname[:2]
            sigungu = cityname[2:]
            
            sido_df = df[df['행정구역(시도)'].str.contains(sido)]
            
            sigungu_df = sido_df[sido_df['행정구역(시군구)'].str.contains(sigungu)]
            
            result = sigungu_df[sigungu_df['연도'] == year]
            
        if len(result) != 1:
            print("길이는 : " + str(len(result)))
            print(result.head())

        return result
        
        

    # %%
    def find_normalizedf_row(do, city):
        row_names = normalized_df.index
        for row_name in row_names:
            if do in row_name and city in row_name:
                return normalized_df.loc[row_name]


    from tqdm import tqdm

    processed_df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop('학령인구(명)')
    error_city_list = []


    for i in tqdm(range(len(df))):
        try:
            parts = df.iloc[i]['행정구역(시도)'].split()
            do = parts[0][:2]
            city = df.iloc[i]['행정구역(시군구)']
            year = df.iloc[i]['연도']

            adj_matrix = find_normalizedf_row(do, city)
            row_list = []
            
                
            for j in range(len(adj_matrix)):
                try:

                    adj_city_name = adj_matrix.index[j]
                    adj_city_weight = adj_matrix.iloc[j]
                    if pd.isna(adj_city_weight):
                        continue
                    
                    

                    row = find_cityname_in_df(df, adj_city_name, year)
                    
                    #tmp_df.insert(row)
                    row[numeric_columns] *= adj_city_weight

                    
                    if(row.empty):
                        error_city_list.append(adj_city_name)
                        continue
                    
                    row_list.append(row)
                except KeyError:
                    print("error 2: " + adj_matrix.index[j])
                    continue
            
                    
            # Sum all the rows in row_list
            new_row = pd.concat(row_list)

            
            new_row = new_row[numeric_columns]
            new_row_sum = new_row.sum()

            

            # Use .loc to avoid SettingWithCopyWarning
            processed_df.loc[processed_df.index[i], numeric_columns] = new_row_sum

        except Exception as e:
            print(f"error 3:  - {str(e)}")
            print(do + " " + city)
            continue

    # procedded_df의 '연도' 컬럼을 반올림해서 int 정수로 바꾸기

    processed_df['연도'] = processed_df['연도'].round().astype(int)
    
    return processed_df


import sys
sys.path.append('../')
import val_generate
import lstm
import numpy as np

distanceList = range(1, 130, 5)
paramsList = np.arange(0.1, 1.0, 0.1)

mapeList = list()


try:
    for i in distanceList:
        for j in paramsList:
            df = parameterTuning(i, j)
            df = val_generate.make_df(df)
            mape = lstm.lstm(df)
            mapeList.append([i, j, mape])
except KeyboardInterrupt:
    print("KeyboardInterrupt")
finally:
    print(mapeList)
    # mapeList를 csv 파일로 저장하기
    mapeList_df = pd.DataFrame(mapeList, columns=['distance', 'param', 'mape'])
    mapeList_df.to_csv('./dataset/mapeList.csv', index=False)
    print("Finish")
"""


# 멀티스레딩을 위한 함수 정의
def thread_function(distance, param):
    df = parameterTuning(distance, param)
    df = val_generate.make_df(df)
    mape = lstm.lstm(df)
    return [distance, param, mape]

from concurrent.futures import ProcessPoolExecutor, as_completed

# 이 부분을 함수로 정의하십시오
def main():
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in distanceList:
            for j in paramsList:
                future = executor.submit(thread_function, i, j)
                futures.append(future)

        try:
            for future in as_completed(futures):
                mapeList.append(future.result())

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            print(mapeList)
            # mapeList를 csv 파일로 저장하기
            mapeList_df = pd.DataFrame(mapeList, columns=['distance', 'param', 'mape'])
            mapeList_df.to_csv('./dataset/mapeList.csv', index=False)
            print("Finish")

# 이 부분이 중요합니다
if __name__ == '__main__':
    main()

"""