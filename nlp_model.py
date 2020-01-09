import os

import numpy as np
import pandas as pd



class NLP_Model():
    def __init__(self):

        #loads in training and test data
        self.trainDF, self.testDF = self.load_data(0.8)
        
        


    def load_data(self, split_prop):
        print('LOADING DATA...')
        fields =['handle','text']
        df = pd.read_csv("tweets_dataset_ex1.csv", usecols=fields)
        
        #shuffle random
        np.random.seed(42)
        df = df.reindex(np.random.permutation(df.index))
        
        #sets how to split data, decimal is hyperparameter
        mask = np.random.rand(len(df)) < split_prop
        trainDF = pd.DataFrame(df[mask])
        testDF = pd.DataFrame(df[~mask])

        print(len(trainDF),len(testDF))

        return trainDF, testDF


    
    def preprocess(self):
        print('PREPROCESSING...')

















##################################################################
##                    HELPER FUNCTIONS                          ##
##################################################################


## Parse function for word embeddings
## Inputs any string
## Returns as a list of cleaned lowered words, all symbols removed
def tweet_word_parse(tweet):
    import re
    return re.sub(r'[^a-zA-Z ]', '',tweet).lower().split(" ")
       
        



if __name__ == '__main__':
    model = NLP_Model()

