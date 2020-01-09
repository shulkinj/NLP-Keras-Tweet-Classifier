import os

import numpy as np
import pandas as pd



class NLP_Model():
    def __init__(self):

        #loads in training and test data
        self.load_data(0.8)
        print("hey guys")


    def load_data(self, p):
        fields =['handle','text']
        df = pd.read_csv("tweets_dataset_ex1.csv", usecols=fields)
        
        #shuffle random
        np.random.seed(42)
        df = df.reindex(np.random.permutation(df.index))
        
        #sets how to split data, decimal is hyperparameter
        mask = np.random.rand(len(df)) < p
        trainDF = pd.DataFrame(df[mask])
        testDF = pd.DataFrame(df[~mask])

        print(len(trainDF),len(testDF))
        
        



if __name__ == '__main__':
    model = NLP_Model()

