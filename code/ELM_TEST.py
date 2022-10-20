#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
from model import ELM
import torch


def get_data_path(boxpath):
    box=os.listdir(boxpath)

    boxs=[]
    for count in range(len(box)):
        im_name=box[count]
        im_path=os.path.join(boxpath,im_name)
        boxs.append(im_path)

    return boxs
def activation(x):
    return 1.0 / (1 + np.exp(-x))

def test(input_data,non, weight01, bias, B, Num):
    h_test = activation(np.dot(input_data, weight01) + bias[:Num, :])
    out_put = np.dot(h_test, B)
    return out_put

def predict(X_test,location_of_model):
    Num = len(X_test)

    X_test = np.stack(X_test)
    # print(X_test.shape)

    pickle_file = open(location_of_model, 'rb')

    model = pickle.load(pickle_file)

    non, weight01, bias, B = model[0], model[1], model[2], model[3]
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
        predict_tensor = torch.tensor(predict)
        labels_tensor = torch.tensor(labels)
        correct = predict_tensor.eq(labels_tensor)
    return correct.sum()/len(predict)

if __name__=='__main__':
    '''labels: 0 - move  
                 1-  still
          inputs is the npy files
      '''
    location_of_model=[
        '/home/prog/code/model/elm/elm_model.pkl',
                       ]
    modelname='model'
   
               ]
    testnumber = [
           1,2,3,4,5]
    X_testdataset = []
    Y_testdataset = []
    movedata,my=[],[]
    stilldata,sy=[],[]

    for testnum in testnumber:
    
        move_data_test_path = '/home//prog/code/data/testdataset/test//move/' + str(testnum)

        # real test data move
        data_test_move = get_data_path(move_data_test_path)

       # for move test dataset
        for i in range(len(data_test_move)):
            d = np.load(data_test_move[i])
            d.resize(d.shape[0]*d.shape[1])
            X_testdataset.append(d)
            Y_testdataset.append(0)
            movedata.append(d)
            my.append(0)


        still_data_test_path = '/home/prog/code/data/testdataset/test//still/' + str(testnum) # real test data still
        data_test_still = get_data_path(still_data_test_path)

        # for still test dataset
        for i in range(len(data_test_still)):
            d = np.load(data_test_still[i])
            d.resize(d.shape[0]*d.shape[1])
            X_testdataset.append(d)
            Y_testdataset.append(1)
            stilldata.append(d)
            sy.append(1)


        for elmmodel in range(len(location_of_model)):
            file = '/home/prog/code/result/ELM_'+modelname[elmmodel]+ '_test_withseparateaccuracy.txt'
            predict_result, out_put1=predict(X_testdataset,location_of_model[elmmodel])
            test_acc=accuracy(predict_result,Y_testdataset)

            predict_result, out_put1 = predict(movedata, location_of_model[elmmodel])
            move_acc = accuracy(predict_result, my)
            predict_result, out_put1 = predict(stilldata, location_of_model[elmmodel])
            still_acc = accuracy(predict_result, sy)

            print(f'\t Test. Acc: {test_acc * 100:.2f}%')
            print(f'\t move_acc. Acc: {move_acc * 100:.2f}%')
            print(f'\t still_acc. Acc: {still_acc * 100:.2f}%')

            with open(file, 'a') as f:
                f.write(f'test dataset {testnum}: \n'
                        f'\t Test. Acc: {test_acc * 100:.2f}%\n'
                        f'\t move_acc. Acc: {move_acc * 100:.2f}%\n'
                        f'\t still_acc. Acc: {still_acc * 100:.2f}%\n\n')










