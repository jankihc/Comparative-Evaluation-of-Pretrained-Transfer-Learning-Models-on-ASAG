import numpy as np
import pandas as pd
from Training.Regression import RegressionAnalysis
from Processing.Stats import Metrics

df_train = pd.read_csv('dataset/answers_with_similarity_score.csv')
df = pd.read_csv('dataset/mistakes_similarity_scores.csv')

ids = df['Unnamed: 0']
print(list(ids)[0])

df_train = df_train[:2273]
train_data = df_train[~df_train['id'].isin(list(ids))]
test_data = df_train[df_train['id'].isin(list(ids))]

