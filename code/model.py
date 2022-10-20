#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:32:46 2019

@author: rongzihan
"""

import numpy as np  # linear algebra
import torch.nn as nn
import pickle
import os





class ELM2(nn.Module):

    def __init__(self, no_of_hidden_nodes1=7000,C=1e-9,no_of_features=10000):
        super(ELM2,self).__init__()
        self.no_of_hidden_nodes1 = no_of_hidden_nodes1
        self.C=C
        self.no_of_features=no_of_features
    def activation(self,x):
        return 1.0 / (1 + np.exp(-x))

    def MPInverse(self,h):
        return np.linalg.pinv(h)

    def hidden_layers(self,flat, no_of_hidden_nodes1, labels):

        no_of_hidden_nodes1 = no_of_hidden_nodes1
        no_of_output = 1

        input_data = flat
        # print(flat[1].shape)
        output = labels

        np.random.seed(2018)
        rnd = np.random.RandomState(4444)
      
        input_layer = input_data
        num = input_data.shape[0]
        bias = np.zeros([num, no_of_hidden_nodes1])

        for i in range(no_of_hidden_nodes1):
            rand_b = rnd.uniform(-1, 1)
            for j in range(num):
                bias[j, i] = rand_b
        bias = np.array(bias)
        weight01 = rnd.uniform(-1, 1, (self.no_of_features, no_of_hidden_nodes1))
        h = self.activation(np.dot(input_layer, weight01) + bias)
        # print('model trained')
        return h, weight01, bias

    def classific_func(self,X_train, labels, C):
        C = C
        I = len(labels)
        h, t, z = self.hidden_layers(X_train, self.no_of_hidden_nodes1, labels)
        sub_former = np.dot(np.transpose(h), h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(h))
        B = np.dot(all_m, labels)
        return B



    def forward(self,X_train,labels):
        h, weight01, bias = self.hidden_layers(X_train, self.no_of_hidden_nodes1, labels)
        B = self.classific_func(X_train, labels, self.C)
        print('training model')
        return h,weight01,bias,B



