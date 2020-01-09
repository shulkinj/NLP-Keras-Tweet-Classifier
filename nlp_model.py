import os

import numpy as np
import pandas as pd

fields =['handle','text']
df = pd.read_csv("tweets_dataset_ex1.csv", usecols=fields)

#shuffle random
np.random.seed(42)
df = df.reindex(np.random.permutation(df.index))
print(len(df))
#sets how to split data
mask = np.random.rand(len(df)) < 0.8
trainDF = pd.DataFrame(df[mask])
testDF = pd.DataFrame(df[~mask])

print(len(trainDF),len(testDF))

print(testDF['handle'])

