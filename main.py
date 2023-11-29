import sys
sys.path.append('../')
import lstm_last
import os

path = './makeDataset/dataset/processedData/61_0.7.csv'

mape, model = lstm_last.lstm(path)
