import os

import numpy as np
import pandas as pd



class NLP_Model():
    def __init__(self):
        ## HyperParameters

        # dict length without max: ~4200
        # probably only necessary if trained on larger datasets
        self.dict_max_length = float('inf')
        

        #loads in training and test data
        self.trainDF, self.testDF = self.load_data(0.8)
        
        
        #constructs dictionary
        #loads training tweets & markings into training data for network
        self.word_index, self.reversed_word_index = self.preprocess(self.dict_max_length) 



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
        print('Train set size:' , len(trainDF))
        print('Test set size:' , len(testDF))

        return trainDF, testDF


    ## Constructs dictionary
    ## Optional dictionary max length by word fequency in training data
    def preprocess(self, max_dict_size=float('inf')):
        print('PREPROCESSING...')
        
        training_tweets = self.trainDF[['text']].values
        
        ## Complete non-stripped dictionary where values are frequency
        ## Here we automatically delete any word containing "http" to get rid of links
        full_dict= {}
        for tweet in training_tweets:
            cleaned= tweet_word_parse(tweet[0])
            for word in cleaned:
                #word already in dic
                if word in full_dict:
                    full_dict[word]+=1
                #prevents links from becoming words
                elif('http' in word):
                    pass
                #adds word to full dic
                else:
                    full_dict[word]=1

        #tuples of words and fequency sorted
        sorted_dict = sorted(full_dict.items(), key =lambda kv:(kv[1], kv[0]))
        #cuts dictionary size to max_dict_size
        sorted_dict = sorted_dict[max(len(sorted_dict)-max_dict_size,0):]
        sorted_dict.reverse()
        
        word_index={}
        
        word_index["<UNK>"]=0
        index=1
        for elt in sorted_dict:
            word_index[elt[0]]=index
            index+=1
        
        
        reversed_word_index = dict([(v,k) for (k,v) in word_index.items()])
        
        return word_index, reversed_word_index














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

