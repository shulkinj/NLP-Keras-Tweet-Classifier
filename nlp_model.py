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
        self.word_index, self.reversed_word_index = self.build_dict(self.dict_max_length) 
        (self.x_train,self.y_train),(self.x_test, self.y_test) = self.preprocess()
      


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
    def build_dict(self, max_dict_size=float('inf')):
        print('BUILDING DICTIONARY...')
        
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
        


        print("Dictionary size: ",len(word_index)," words ")
        return word_index, reversed_word_index
    


    ## Returns an encoding vector for inputted tweet corresponding to word_index
    def encode(self,tweet):
        split_up = tweet_word_parse(tweet)
        encoding = []
        for word in split_up:
            if(word in self.word_index):
                encoding.append(self.word_index[word])
            else:
                encoding.append(0)
        return encoding
    

    ## Outputs train and test inputs and outputs for use in model
    def preprocess(self):
        print("PREPROCESSING...")
        ## Encodes all train and test tweets into tweet vectors, words corresponding to indices
        x_train = []
        x_test = []
        train_tweets=self.trainDF[['text']].values
        test_tweets=self.testDF[['text']].values
        for tweet in train_tweets:
            x_train.append(self.encode(tweet[0]))
        for tweet in test_tweets:
            x_test.append(self.encode(tweet[0]))

        y_train = []
        y_test = []
        train_markings=self.trainDF[['handle']].values
        test_markings=self.testDF[['handle']].values
        for marking in train_markings:
            if marking[0]=='realDonaldTrump':
                y_train.append(0)
            elif marking[0]=='HillaryClinton':
                y_train.append(1)

        for marking in test_markings:
            if marking[0]=='realDonaldTrump':
                y_test.append(0)
            elif marking[0]=='HillaryClinton':
                y_test.append(1)


        return (x_train,y_train) , (x_test,y_test)

    













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

