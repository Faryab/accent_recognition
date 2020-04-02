#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:52:15 2020

@author: emilykmp
"""


# fyi - I had to manually install/upgrade many packages to get this part to run

from __future__ import absolute_import, division, print_function, unicode_literals

import re
import random
import pylab
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd

from scipy.io import wavfile as wav
from python_speech_features import logfbank, mfcc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling1D, Dropout
from keras.optimizers import Adam



import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models



def low_med_high(val, b1, b2):
    # take the continuous measures
    # bin them into groups of low, medium, and high
    # the original scale was 1-5
    #   val: original data
    #   b1: 1st boundary for binning
    #   b2: 2nd boundary for binning
    
    if val < b1:
        out = 0
    elif val < b2:
        out = 1
    else:
        out = 2
    return out


def process_doc(data_path, b1, b2):
    # the goal is to transform the documentation file into a convenient data structure
    #   data_path: where the documentation is stored
    #   b1: 1st boundary for binning
    #   b2: 2nd boundary for binning
    emo_dict = {}
    # this is where the labels and filenames are stored
    file = open(data_path + 'DocumentationEma.txt', 'r') 
    for line in file.readlines():
        if line[0] == "#":
            # split by spaces
            items = re.split("\s+", line.strip())
            
            # extract audio features, both mfb and mfcc
            (rate,sig) = wav.read(data_path + 'wav/' + items[2])
            mfb = logfbank(sig,rate)
            mfc = mfcc(sig,rate) 
            
            # what is the emotion?
            # change emotion categories into numbers
            p = re.compile('(angry|happy|neutral|sad)')
            name = {}
            name['angry'] = 0
            name['happy'] = 1
            name['neutral'] = 2
            name['sad'] = 3
            
            # valence, activation, dominance
            # bin them into low, medium, and high
            val = low_med_high(float(items[3]), b1, b2)
            act = low_med_high(float(items[4]), b1, b2)
            dom = low_med_high(float(items[5]), b1, b2)
             
            # create a dictionary, indexed by filename
            #   disc_emo: the discrete emotion label (numeric categories)
            #   val: binned valence
            #   act: binned activation
            #   dom: dominance
            #   mfb: Mel Filterbanks (log Mel)
            #   mfcc: Mel Frequency Cepstral Coefficients
            emo_dict[items[2]] = {'disc_emo': name[p.search(items[2]).group(0)], 'val': val, 'act': act, 'dom': dom, 'mfb': mfb, 'mfcc': mfc}
            
    file.close()
    return emo_dict


def process_data(partition, feat_type, emo_type, batch=False):
    
    # get data into correct format
    x_data = [x for x in partition[feat_type].to_list()]
    y_labels = [x for x in partition[emo_type].to_list()]    
    
    if batch:
        
        # create four batches (arbitrary choice)
        x_data_b = [x_data[0::4], x_data[1::4], x_data[2::4], x_data[3::4]]
        y_labels_b = [y_labels[0::4], y_labels[1::4], y_labels[2::4], y_labels[3::4]]
        
        # zero pad the time series
        # everything in a single batch should be the same length
        dataset = []
        for i, xt in enumerate(x_data_b):
            lens = [len(x) for x in xt]
            max_len = max(lens)
            
            data = np.zeros((len(xt), max_len, num_feats))
            labels = np.zeros((len(xt), max_len)) 
            for j, n in enumerate(data):
                data[j,:len(xt[j]),:] = xt[j]
                labels[j,:len(xt[j])] = y_labels_b[i][j]
    
            dataset.append(tf.data.Dataset.from_tensor_slices((data, labels)))
        
    else:
    
        # zero pad the time series
        # everything in a single batch should be the same length
        lens = [len(x) for x in x_data]
        max_len = max(lens)
        
        # this will hold the data and the labels
        #   data: num instances x maximum length of instances x number of features
        #   labels: number of instances x maximum length of instances (one label for each measurement)
        data = np.zeros((len(x_data), max_len, num_feats))
        labels = np.zeros((len(x_data), max_len)) 
        print (data.shape)
        
        # update data and labels
        for j, n in enumerate(data):
            # add each obersvation
            # there will be zeros if the utterance length is less than the maximum length
            data[j,:len(x_data[j]),:] = x_data[j]
            # add the labels
            labels[j,:len(x_data[j])] = y_labels[j]
    
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    
    return dataset


def process_data_cnn(partition, me, st, feat_type, emo_type, seq_len):
    
    
    # data
    x_data = [(x-me)/st for x in partition[feat_type].to_list()]
    y_labels = [x for x in partition[emo_type].to_list()] 
    
    # zero pad all to this len
    for i, x in enumerate(x_data):
        template = np.zeros((seq_len,num_feats))
        template[:len(x)] = x
        x_data[i] = template
        
    # create data and labels
    data = np.zeros((len(x_data), seq_len, num_feats))
    for i, x in enumerate(x_data):
        data[i,:len(x),:] = x
    
    labels = np.array(y_labels).reshape(len(y_labels),1)
    
    return data, labels


#%% the main code
# note: this is what we used in the feature selection demo!

if __name__ == "__main__":
    # boundaries to discretize continuous (scale:1-5) valence/activation into low/med/high
    b1 = 2.5
    b2 = 3.5
    
    # for ease, encode how many models are required by each type of data
    model_spec = {}
    model_spec['disc_emo'] = 4      # discrete emotion (angry, happy, neutral, sad)
    model_spec['val'] = 3
    model_spec['act'] = 3
    model_spec['dom'] = 3
    
    # if repeating experiment, how many times?
    # note: if there is randomness, this helps you get a sense for how
    #       your system is performing on average vs. one lucky/unlucky run
    num_it = 10
    
    # process documentation file
    data_path = 'data/'
    emo_dict = process_doc(data_path, b1, b2)
    
    # create data frame
    df = pd.DataFrame(emo_dict).T
    
    #%% make some choices
    
    # type of emotion
    # disc_emo: discrete emotion (angry, happy, neutral, sad)
    # val: valence (high, medium, low)
    # act: activation (high, medium, low)
    # dom: dominance (high, medium, low)
    emo_type = 'act'
    
    # type of feature
    feat_type = 'mfb'
    
    # number of models
    num_mod = model_spec[emo_type]
    
    # number of features
    num_feats = df.mfb[0].shape[1]
    
    #%% prepare
    
    # create train/test split 
    # I am going to do this once and then use the splits in the remainder
    # This avoids the problem of performance changing because the test set is different
    train, test = train_test_split(df, test_size=0.2)
    
    # create a validation partition, like in the cnn example!
    train_size = train.shape[0]
    perc = 0.80
    tr_ind = random.sample(list(np.arange(1,train_size)), int(np.floor(perc*train_size)))
    val_ind = list(set(np.arange(0,train_size)) - set(tr_ind))
    
    val = train.iloc[val_ind]
    train = train.iloc[tr_ind]
    
    #%% Create a lstm
    
    # fix random seed
    tf.random.set_seed(26)
    
    # input dimension is the number of mfbs
    dim = num_feats
    
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input size that is 40-dim (dim)
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=dim, output_dim=64))
    
    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128, input_dim=dim, return_sequences=False))
    
    # Add a Dense layer with 10 units.
    model.add(layers.Dense(num_mod))
    
    model.summary()

    # compile the model
    # using:
    #   - the adam optimizer (see for more info: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
    #   - categorical cross entropy (also know as: softmax!)
    #   - accuracy as the metric (this works since the classes are relatively balanced)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    #%% preprocess data (one batch!)
    
    # create datasets for train, validation, and test
    dataset_train = process_data(train, feat_type, emo_type, batch=False)
    dataset_val = process_data(val, feat_type, emo_type, batch=False)
    dataset_test = process_data(test, feat_type, emo_type, batch=False)
    
    # how many epochs total?
    num_epochs = 10

    # this will hold the performance
    perf_time = np.zeros((num_epochs, 3))
    
    # train it once
    model.fit(dataset_train.shuffle(100), epochs=1, verbose=1, 
                        validation_data=dataset_val, validation_steps=100)
    
    # check how it is doing on train, test, and validation data
    new = [model.evaluate(dataset_train)[1], 
               model.evaluate(dataset_val)[1], 
               model.evaluate(dataset_test)[1]]
    
    # store this (this is the starting point)
    perf_time[0,:]=new
    
    
    #%% train for the remainder of epochs
    for epoch in np.arange(1,num_epochs):
        # train an epoch at a time, visualize as we go!
        model.fit(dataset_train.shuffle(100), epochs=1, verbose=1, 
                        validation_data=dataset_val, validation_steps=100)
        
        # check the performance on train/test/val
        # the model.evaluate function returns the loss (position 0) and the performance (position 1)
        new = [model.evaluate(dataset_train)[1], 
               model.evaluate(dataset_val)[1], 
               model.evaluate(dataset_test)[1]]
        
        # add to performance
        perf_time[epoch,:]=new
        
        # visualize
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,0],'b', label='train')
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,1],'r', label='validation')
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,2],'g', label='test')
        plt.legend(loc='upper left')
        plt.show()
        
    print ("final test performance: %f")
        
    
    #%% preprocess data (batching)
        
    # calculate normalization statistics
    x_train_all = [y for x in  train[feat_type].to_list() for y in x]
    
    dataset_train = process_data(train, feat_type, emo_type, batch=True)
    dataset_val = process_data(val, feat_type, emo_type, batch=True)
    dataset_test = process_data(test, feat_type, emo_type, batch=True)
    
    # train on the batch (note: this is slow)
    max_epoch = 5
    for e in np.arange(0,max_epoch):
        for i, d in enumerate(dataset_train):
            print ("epoch %d, batch %d" %(e, i))
            model.fit(d, epochs=1, verbose=1, 
                        validation_data=dataset_val[i],
                        validation_steps=100)
    score = model.evaluate(dataset_test, batch_size=16)
    print ("test accuracy: %.2f" %score[1])
            
    #%% what if we use a cnn instead?
    
    # calculate normalization statistics
    x_train_all = [y for x in  train[feat_type].to_list() for y in x]
    me = np.mean(x_train_all,axis=0)
    st = np.std(x_train_all,axis=0)
    
    seq_len = max(max([len(x) for x in train[feat_type].to_list()]),
                  max([len(x) for x in test[feat_type].to_list()]),
                  max([len(x) for x in val[feat_type].to_list()]))
    
    data_train, label_train = process_data_cnn(train, me, st, feat_type, emo_type, seq_len)
    data_test, label_test = process_data_cnn(test, me, st, feat_type, emo_type, seq_len)
    data_val, label_val = process_data_cnn(val, me, st, feat_type, emo_type, seq_len)
            
    # create a linear stack of layers
    # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    model = models.Sequential()
    # the first layer is a convolutional layer:
    #   - 16 filters
    #   - kernel width of 64
    #   - relu (rectified linear unit) activation
    #   - input shape (from the data)
    #   - see for more options: https://www.tensorflow.org/api_docs/python/tf/keras/layers/
    
    model.add(layers.Conv1D(16, 64, strides=1, 
        dilation_rate=1, activation='relu', 
        input_shape=(seq_len, num_feats), name="layer1"))
    
    # then max pooling 
    model.add(layers.GlobalMaxPooling1D())
    
    # flatten
    model.add(layers.Flatten())
    # dense
    model.add(layers.Dense(64, activation='relu'))
    # the final number of outputs need to match the number of classes (one for each class)
    model.add(layers.Dense(num_mod))
    model.add(layers.Dropout(0.5))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  
    
    # train it for 15 epochs
    history = model.fit(data_train, label_train, 
                        batch_size=16, epochs=15, verbose=1, 
                        validation_data=(data_val, label_val))
    
    # how well does it do on the test data?
    score = model.evaluate(data_test, label_test, batch_size=16)
    print ("test accuracy: %.2f" %score[1])
        
        
        
        
        
        
        
        
        