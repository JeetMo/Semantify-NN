#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup GTSRB data

@author: smartlily
"""

import os
import tarfile
import urllib.request
import zipfile

from PIL import Image
import numpy as np

import random
random.seed(123)
np.random.seed(123)


## Earlier loading data function: load both training data and test data at once
def load_data():
    
    print("Checking if data exist:")

    ########### check data exist ###########
    
    if not os.path.exists("data"):
        print("data not exist")
        os.mkdir("data")

    # the miniplace contains two folders: images and objects    
    if not os.path.exists("data/GTSRB"):
        print("GTSRB not exist")
        os.mkdir("data/GTSRB")
    
    if not os.path.exists("data/GTSRB/Final_Training/"):
        if not os.path.exists("data/GTSRB_Final_Training_Images.zip"):
            print("Downloading GTSRB_Final_Training_Images.zip ......")         
            urllib.request.urlretrieve("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip","data/GTSRB_Final_Training_Images.zip")
            
        print("Extracting GTSRB_Final_Training_Images.zip to data/GTSRB/Final_Training ......")   
        with zipfile.ZipFile("data/GTSRB_Final_Training_Images.zip", "r") as zip_ref:
            zip_ref.extractall("data/")
    else:
        print("GTSRB training data exist!")

    if not os.path.exists("data/GTSRB/Final_Test/"):
        if not os.path.exists("data/GTSRB_Final_Test_Images.zip"):
            print("Downloading GTSRB_Final_Test_Images.zip ......")         
            urllib.request.urlretrieve("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip","data/GTSRB_Final_Test_Images.zip")
        
        print("Extracting GTSRB_Final_Test_Images.zip to data/GTSRB/Final_Test/ ......")  
        with zipfile.ZipFile("data/GTSRB_Final_Test_Images.zip", "r") as zip_ref:
            zip_ref.extractall("data/")
    else:
        print("GTSRB test data exist!")


    ## check the extract is successful
    if not os.path.exists("data/GTSRB/Final_Training/Images/00000/00000_00000.ppm"):
        print("!!!!! the GTSRB data is not extracted successfully !!!!")
        raise ValueError("the GTSRB data is not extracted successfully")
    
    #########################################
    
    # links to be added later: use lily's github
    # get the train and test data labels and path:
    if not os.path.exists("data/GTSRB/train.txt"):
        #urllib.request.urlretrieve("https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt","data/miniplace/train.txt")
        urllib.request.urlretrieve("https://lilyweng.github.io/exp/gtsrb/train.txt","data/GTSRB/train.txt")
    if not os.path.exists("data/GTSRB/test.txt"):
        #urllib.request.urlretrieve("https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt","data/miniplace/val.txt")
        urllib.request.urlretrieve("https://lilyweng.github.io/exp/gtsrb/test.txt","data/GTSRB/test.txt")
    
    
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    val_fraction = 0.1
    n_test = 12630
    X_test = np.zeros([n_test,28,28,3])
    y_test = np.zeros([n_test], dtype = 'uint8')
    
    num_class = 43
    
    n_train = 39209
    X_train = np.zeros([n_train,28,28,3])
    y_train = np.zeros([n_train], dtype = 'uint8')
    
    # read in all the training data:
    # it'll have memory error (on Asus laptop). 
    # Need to use server or either miniplace provided dataloader script to train
    with open("data/GTSRB/train.txt") as fp:
        cnt = 0
        for line in fp:
            filename, label, _ = line.split(" ")
            #print("file: {} label: {}".format(filename,label))
            img = Image.open('data/'+filename).resize((28,28))
            img = np.asarray(img)
            #print("img shape = {}, img max = {}, img min = {}".format(img.shape,np.max(img),np.min(img)))
             
            # normalize data: original data 0 to 255, now normalized to -0.5 to 0.5
            X_train[cnt] = (img/255) - 0.5
            y_train[cnt] = int(label)
            cnt += 1
    
    assert cnt == n_train, "cnt is not equal to n_train. Problem with the train.txt"
   
    ## prepare to split the training data
    n_val = int(n_train*val_fraction)
    idxs = np.random.permutation(n_train)
    
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]
    
    # split training data to val and train
    X_val = X_train[val_idxs]
    X_train = X_train[train_idxs]
    # expand the labels to 1-hot vector: n_train*n_class
    y_val = np.eye(num_class)[y_train[val_idxs]]
    y_train = np.eye(num_class)[y_train[train_idxs]]
   
    print("---- finish loading miniplace training data ----")
    print("---- data range: {} to {} ----".format(np.min(X_val),np.max(X_val)))

        
    # read in all the test data as test data (because miniPlace didn't release test data label)   
    with open("data/GTSRB/test.txt") as fp:
        cnt = 0
        for line in fp:
            filename, label, _ = line.split(" ")
            #print("file: {} label: {}".format(filename,label))
            img = Image.open('data/'+filename).resize((28,28))
            img = np.asarray(img)
            #print("img shape = {}".format(img.shape))
            # normalize data: original data 0 to 255, now normalized to -0.5 to 0.5
            X_test[cnt] = (img/255) - 0.5
            y_test[cnt] = int(label)
            cnt += 1
            
        y_test = np.eye(num_class)[y_test]
    
    
        print("---- finish loading miniplace test data ----")
        print("---- data range: {} to {} ----".format(np.min(X_test),np.max(X_test)))
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class GTSRB():
    def __init__(self):
       
        X_train, y_train, X_val, y_val, X_test, y_test = load_data()
        self.test_data = X_test
        self.test_labels = y_test
        self.train_data = X_train
        self.train_labels = y_train
        self._ntrain = 39209
        self.num_class = 43




if __name__ == "__main__":
    
    
    G = GTSRB()

    print("shape of test_data: {}, test_labels: {}".format(G.test_data.shape,G.test_labels.shape)) 
    print("shape of X_train = {}, y_train = {}".format(G.train_data.shape,G.train_labels.shape))
  
    

    
    

    
