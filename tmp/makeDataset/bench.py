
import pandas as pd


filePath = './dataset/adjacency_matrix.csv'
distance_matrix = pd.read_csv(filePath, index_col=0)



distance_threshold = 126
local_params = 0.6
adj_params = 1 - local_params


dm_with_threshhold = distance_matrix[distance_matrix < distance_threshold]



dm_with_threshhold = dm_with_threshhold.replace(0, 1)
dm_with_threshhold = 1 / dm_with_threshhold



import numpy as np


row_sums = dm_with_threshhold.sum(axis=1) - 1



normalized_df = dm_with_threshhold.div(row_sums, axis=0) * adj_params



for i in range(len(normalized_df)):
    if normalized_df.iloc[i, i] != np.inf:
        normalized_df.iloc[i, i] = local_params
    else:
        normalized_df.iloc[i, i] = 1



result = normalized_df.sum(axis=1)

result[result - 1 > 0.00001]


normalized_df.to_csv('./dataset/distance_normalize_v2.csv')

df = pd.read_csv('./dataset/no_NaN_dataset_final.csv')

def find_cityname_in_df(df, cityname, year):
    sido = cityname[:2]
    sigungu = cityname[2:]
    
    result = df[(df['행정구역(시도)'].str.contains(sido)) & 
                (df['행정구역(시군구)'] == sigungu) & 
                (df['연도'] == year)]
    
    if len(result) != 1:
        result = df[(df['행정구역(시도)'].str.contains(sido)) & 
                (df['행정구역(시군구)'].str.contains(sigungu)) & 
                (df['연도'] == year)]

    if len(result) != 1:
        print("길이는 : " + str(len(result)))
        print(result.head())

    return result


def find_normalizedf_row(do, city):
    matching_rows = normalized_df.index[normalized_df.index.str.contains(do) & normalized_df.index.str.contains(city)]
    if matching_rows.empty:
        return pd.DataFrame()
    else:
        return normalized_df.loc[matching_rows].iloc[0]



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
        

        weight = 0

        
        #tmp_df = pd.DataFrame(columns=df.columns)
            
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
        
        if i == 0:
            print(new_row.head())
        
        new_row = new_row[numeric_columns]
        new_row_sum = new_row.sum()
        
        if i == 0:
            print(new_row_sum.head())

        #if i == 1:
            #print(yearlist)
            #print(new_row['연도'])
        

        # Use .loc to avoid SettingWithCopyWarning
        processed_df.loc[processed_df.index[i], numeric_columns] = new_row_sum

    except Exception as e:
        print(f"error 3:  - {str(e)}")
        print(do + " " + city)
        continue



processed_df['연도'] = processed_df['연도'].round().astype(int)

processed_df.to_csv('./dataset/weighted_dataset_laewon.csv')
