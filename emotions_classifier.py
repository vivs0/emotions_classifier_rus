#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:32:12 2018

@author: dns
"""
import pandas as pd
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras import regularizers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical

import pymorphy2
import re
from enum import Enum


class vect_types(Enum):
    TF_IDF = 1
    COUNT  = 2

class transform_types(Enum):
    FREQ = 1

class preprocess_options(Enum):
    SEQ = 1
    BOW = 2

class emotion_classifier():
    def __init__(self):
        pass
    
    def __apply_class2mood(self):
        class2mood = {
            (1,1): 0,#'excited,delighted,aroused,astonished',
            (1,0): 1,#'calm,relaxed,content, friendly',
            (0,1): 2,#'angry annoyed, frustrated, disguted',
            (0,0): 3#'depressed, bored, sad, gloomy'
            }
        res = []
        for idx,row in self.__corpus.iterrows():
            val = row['val_class']
            aro = row['aro_class']
            res.append(class2mood[(val,aro)])
        self.__corpus['multiclass'] = res
    
    def __load_corpus(self, filepath):
        self.__corpus = pd.read_csv(filepath).dropna(axis = 0)
        self.__corpus['val_class'] = self.__corpus['Val.W'].apply(lambda x: 1 if x>3.0 else 0)
        self.__corpus['aro_class'] = self.__corpus['Aro.W'].apply(lambda x: 1 if x>3.0 else 0)
        self.__apply_class2mood()
    
    def __vectorize(self, vect_type):
        vd = {
                vect_types.TF_IDF: TfidfVectorizer(ngram_range = (1,3), min_df = 3),
                vect_types.COUNT: CountVectorizer(ngram_range = (1,3), min_df = 3)
              }
        self.__vect = vd[vect_type].fit(self.__corpus.processed_ru)
    
    def __transform_data(self, transform_type):
        if transform_type is transform_types.FREQ:
            self.__vectorize(vect_types.COUNT)
        self.__feats = self.__vect.transform(self.__corpus.processed_ru)
        self.__labels = self.__corpus.multiclass

    def __eval_model(self, y_train,y_test,y_train_pred,y_test_pred):
        class_names = ['excited,delighted,aroused,astonished',
            'calm,relaxed,content, friendly',
            'angry annoyed, frustrated, disguted',
            'depressed, bored, sad, gloomy']
        print('train scores\n')
        print(classification_report(y_train, y_train_pred, target_names = class_names))
        print('test scores\n')
        print(classification_report(y_test, y_test_pred, target_names = class_names))
    
    def __fit_classifier(self):
        X_train, X_test, y_train, y_test = train_test_split(self.__feats, self.__labels, test_size=0.2)
        self.__model = RidgeClassifier(alpha = 100, class_weight = 'balanced').fit(X_train, y_train)
        y_train_pred = self.__model.predict(X_train)
        y_test_pred = self.__model.predict(X_test)
        self.__eval_model(y_train,y_test,y_train_pred,y_test_pred)

    def make_classifier(self):
        self.__load_corpus('emo_bank_ru.csv')
        self.__transform_data(transform_types.FREQ)
        self.__fit_classifier()
    
    def __make_sequences(self, max_length):
        t = Tokenizer()
        t.fit_on_texts(self.__corpus.processed_ru.tolist())
        self.__vocab_size = len(t.word_index) + 1
        encoded_docs = t.texts_to_sequences(self.__corpus.processed_ru)
        feats = sequence.pad_sequences(encoded_docs, maxlen=max_length)
        self.__tokenizer = t
        return feats, to_categorical(self.__corpus.multiclass)
    def __make_single_sequence(self, text, max_length):
        encoded_doc = self.__tokenizer.texts_to_sequences([text])
        feats = sequence.pad_sequences(encoded_doc, maxlen=max_length)
        return feats
    
    def __create_net(self, max_length):
        model = Sequential()
        model.add(Embedding(self.__vocab_size, 50, 
                            input_length=max_length,
                            embeddings_regularizer = regularizers.l2(1e-5)))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=15, kernel_size=3, padding='same', activation='sigmoid'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.2))
        model.add(LSTM(10, activation = 'tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.__nnet = model
    
    def __prepare_net_data(self, preprocess_option, max_length):
        preprocess = {
                preprocess_options.SEQ: self.__make_sequences(max_length),
                preprocess_options.BOW: self.__vectorize(vect_type = vect_types.COUNT)
                }
        self.__netin, self.__netout = preprocess[preprocess_option]
    
    def __train_net(self):
        X_train, X_test, y_train, y_test = train_test_split(self.__netin, self.__netout, test_size=0.2)
        class_weight = compute_class_weight('balanced'
                                               ,[0,1,2,3]
                                               ,self.__corpus.multiclass.apply(int).tolist())
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True, monitor = 'val_acc')
        self.__nnet.fit(X_train, y_train, epochs=100, batch_size=500, validation_data = [X_test,y_test], callbacks=[checkpointer], class_weight = class_weight)
    
    def make_neural_net(self):
        self.__max_length = 30
        self.__load_corpus('emo_bank_ru.csv')
        self.__prepare_net_data(preprocess_options.SEQ, self.__max_length)
        self.__create_net(self.__max_length)
        self.__train_net()
        
    def __clean_text(self, text):
        morph = pymorphy2.MorphAnalyzer()
        text = re.sub(r'[1-9a-zA-Z\^\*\/\$\@\_\"\\n\)\(\.\,\:\;\!\[\]]',' ',text)
        tokens = [morph.parse(w)[0].normal_form for w in gensim.utils.simple_preprocess(text, deacc = True, min_len = 1) if len(w)>2]
        return ' '.join(tokens)

    def __transform_cleaned_text(self, cleaned_text):
        return self.__vect.transform(cleaned_text)
    
    def run_classifier(self, text):
        class_names = {
                0: 'excited,delighted,aroused,astonished',
                1: 'calm,relaxed,content, friendly',
                2: 'angry annoyed, frustrated, disguted',
                3: 'depressed, bored, sad, gloomy'
                }
        cleaned_text = self.__clean_text(text)
        feats = self.__transform_cleaned_text([cleaned_text])
        pred_class = self.__model.predict(feats)
        print('text: %s\nclassified as %s'%(text, class_names[pred_class[0]]))
        return class_names[pred_class[0]]
    def run_neural_network(self, text):
        class_names = {
                0: 'excited,delighted,aroused,astonished',
                1: 'calm,relaxed,content, friendly',
                2: 'angry annoyed, frustrated, disguted',
                3: 'depressed, bored, sad, gloomy'
                }
        cleaned_text = self.__clean_text(text)
        feats = self.__make_single_sequence(cleaned_text,self.__max_length)
        pred_class = self.__nnet.predict(feats)
        print('text: %s\nclassified as %s'%(text, class_names[np.argmax(pred_class[0])]))
        return class_names[np.argmax(pred_class[0])]
    
if __name__ == '__main__':
    ec = emotion_classifier()
    ec.make_neural_net()
    ec.run_neural_network('Наконец-то лето!')
    ec.run_neural_network('Ну и кто они после этого?!')
    ec.run_neural_network('Только приехали и уже уезжают. Уже скучаааю')
    ec.run_neural_network('Неча на зеркало пенять коль рожа крива')