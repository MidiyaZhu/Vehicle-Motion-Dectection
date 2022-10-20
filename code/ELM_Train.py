#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:32:46 2019

@author: rongzihan
"""

import torch
import numpy as np  # linear algebra
import pickle
import os
from model import ELM2
import time
def get_data_path(boxpath):
    box=os.listdir(boxpath)

    boxs=[]
    for count in range(len(box)):
        im_name=box[count]
        im_path=os.path.join(boxpath,im_name)
        boxs.append(im_path)

    return boxs


def test(input_data,non, weight01, bias, B, Num):
    h_test = ELM_model.activation(np.dot(input_data, weight01) + bias[:Num, :])
    out_put = np.dot(h_test, B)
    return out_put

def predict(X_test,non, weight01, bias, B ):
    Num = len(X_test)

    X_test = np.stack(X_test)
 
    out_put1 = test(X_test,non, weight01, bias, B, Num)

    thershold = 0.4  #originally be 0.5

    predict_result = []
    for i in range(len(out_put1)):
        if out_put1[i] <=thershold:# move
            predict_result.append(0)

        if out_put1[i] > thershold:#still

            predict_result.append(1)

    return predict_result, out_put1

def accuracy(predict,labels):
    if len(predict) != len(labels):
        return -1
    else:
        predict_tensor=torch.tensor(predict)
        labels_tensor=torch.tensor(labels)
        correct = predict_tensor.eq(labels_tensor)
    return correct.sum()/len(predict)

def save_elm(path,h,weight01,bias,B):

    model = {}
    model[0] = h
    model[1] = weight01
    model[2] = bias
    model[3] = B
  
    pickle_file = open(path, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__=='__main__':
    '''labels: 0 - move  
               1-  still
        inputs is the npy files
    '''

    data_path_move = '/data/npy/train/move/'
    data = get_data_path(data_path_move)
    X_train = []
    Y_train = []
    for i in range(len(data)):
        d = np.load(data[i])
        dd = np.concatenate((d[0], d[1]), axis=1)
        dd.resize(20000)
        X_train.append(dd)
        Y_train.append(0)
    data_path_still = '/data/npy/train/still/'
    data1 = get_data_path(data_path_still)

    for i in range(len(data1)):
        d1 = np.load(data1[i])
        dd1 = np.concatenate((d1[0], d1[1]), axis=1)
        dd1.resize(20000)
        X_train.append(dd1)
        Y_train.append(1)

    no_of_hidden_nodes1=[
         7592]
  
    C = [
     1e-9]
    for hid in range(len(no_of_hidden_nodes1)):
        file = '/home/prog/code/log/log_ELM_'+str(C[hid])+'_'+str(no_of_hidden_nodes1[hid])+'.txt'
        with open(file, 'a') as f:

            f.write('test now\n')
        start_time = time.time()
        ELM_model= ELM2(no_of_hidden_nodes1=no_of_hidden_nodes1[hid],C=C[hid],no_of_features=20000)
        non, weight01, bias, B=ELM_model(np.stack(X_train),Y_train)
        end_time = time.time()

        predict_result, out_put1 = predict(X_train,   non, weight01, bias, B)
        train_acc = accuracy(predict_result, Y_train)


        print(f'\tTrain Acc: {train_acc * 100:.2f}%')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
        with open(file, 'a') as f:

            f.write(f'\tTrain Acc: {train_acc * 100:.2f}%\n'
                    f'Epoch Time: {epoch_mins}m {epoch_secs}s\n')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)



        '''SAVE MODEL'''
        path='/home//prog/code/model/elm/elm_model_'+str(C[hid])+'_'+str(no_of_hidden_nodes1[hid])+'.pkl'
        save_elm(path, non, weight01, bias, B)
        print('save model done!')
