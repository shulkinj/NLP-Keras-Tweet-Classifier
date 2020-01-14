from __future__ import absolute_import, division, print_function, unicode_literals
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Input, Dense, Reshape, Dot, Embedding


class NLP_Model():
    def __init__(self):
        ## HyperParameters

        # dict length without max: ~4200
        # probably only necessary if trained on larger datasets
        dict_max_length = float('inf')
        # word2vec parameters
        WINDOW_SZ = 4
        EMBEDDING_DIM = 20
        W2V_EPOCHS = 10000
        


        #loads in training and test data
        self.trainDF, self.testDF = self.load_data(0.8)
        
                #constructs dictionary
        #loads training tweets & markings into training data for network
        self.word_index, self.reversed_word_index, self.sorted_dict = self.build_dict(dict_max_length) 
        (self.x_train,self.y_train),(self.x_test, self.y_test) = self.preprocess()

        self.vocab_size = len(self.word_index) 

        self.word2vec(WINDOW_SZ , EMBEDDING_DIM, W2V_EPOCHS)
      


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
        return word_index, reversed_word_index, sorted_dict
    


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
            elif marking[0]=='Hillar)Clinton':
                y_test.append(1)


        return (x_train,y_train) , (x_test,y_test)

    ## Trains word embeddings
    ## Outputs embeddings
    def word2vec(self, WINDOW_SZ, EMBEDDING_DIM, W2V_EPOCHS):
        print("WORD2VEC...")

        valid_size = 20 #random word set to evaluate similarity
        valid_window = 500 #pick samples in 500 most common words
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        vocab_size = self.vocab_size


        ## skipgram set up
        sampling_table= sequence.make_sampling_table(vocab_size, sampling_factor=0.01)

        skipgrams = [sequence.skipgrams(tweet, vocab_size,window_size=WINDOW_SZ, sampling_table=sampling_table)
                    for tweet in self.x_train]

        couples , labels = skipgrams[0][0] , skipgrams[0][1]
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        
        ## Functional API model

        #input layers take in target and context word as ints
        input_target = layers.Input((1,))
        input_context = layers.Input((1,))
        
        #embedding layer then transpose vectors to take dot prod
        embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=1, name='embedding')
        
        target = embedding(input_target)
        target = Reshape((EMBEDDING_DIM, 1))(target)
        context= embedding(input_context)
        context= Reshape((EMBEDDING_DIM, 1))(context)

        #cosine similarity to be used in validation model
        similarity = Dot( axes=0, normalize= True)


        #dot product layers to measure similarity
        dot_product = Dot(axes=1)([target,context])
        dot_product = Reshape((1,))(dot_product)
        #sigmoid output layer
        output = Dense(1, activation = 'sigmoid')(dot_product)

        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        
        #cosine similarity to be used in validation model
        similarity = Dot( axes=1, normalize= True)([target,context])
        validation_model = Model(inputs=[input_target,input_context], outputs=similarity)
        

        reversed_word_index=self.reversed_word_index
        ## Helper class for validating Word2Vec while training
        class SimilarityCallback:
            def run_sim(self):
                for i in range(valid_size):
                    valid_word = reversed_word_index[valid_examples[i]]
                    top_k = 8 #num of nearest neighbors
                    sim = self._get_sim(valid_examples[i])
                    nearest = (-sim).argsort()[1 : top_k+1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reversed_word_index[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

            @staticmethod
            def _get_sim(valid_word_idx):
                sim = np.zeros((vocab_size,))
                in_arr1 = np.zeros((1,))
                in_arr2 = np.zeros((1,))
                for i in range(vocab_size):
                    in_arr1[0,] = valid_word_idx
                    in_arr2[0,] = i
                    out = validation_model.predict_on_batch([in_arr1,in_arr2])
                    sim[i] = out
                return sim
        
        sim_cb = SimilarityCallback()

        arr_1 = np.zeros((1,))
        arr_2 = np.zeros((1,))
        arr_3 = np.zeros((1,))
        for cnt in range(W2V_EPOCHS):
            idx = np.random.randint(0, len(labels)-1)
            arr_1[0,] = word_target[idx]
            arr_2[0,] = word_context[idx]
            arr_3[0,] = labels[idx]
            loss = model.train_on_batch([arr_1, arr_2], arr_3)
            if cnt % 100 == 0:
               print("Iteration {}, loss={}".format(cnt, loss))
            if cnt % 500 == 0:
               sim_cb.run_sim()





       


##################################################################
##                    HELPER FUNCTIONS                          ##
##################################################################


## Parse function for word embeddings
## Inputs any string
## Returns as a list of cleaned lowered words, all symbols removed
def tweet_word_parse(tweet):
    import re
    return re.sub(r'[^a-zA-Z ]', '',tweet).lower().split(" ")

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

        



if __name__ == '__main__':
    model = NLP_Model()

