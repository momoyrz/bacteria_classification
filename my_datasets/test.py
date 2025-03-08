import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
df = pd.read_csv('/home/ubuntu/qujunlong/data/tryy/train_set.csv')
print(df.head())