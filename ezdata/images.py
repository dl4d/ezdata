#
# Image wrapper for both classification, segmentation and encoder dataset
#
import os
from PIL import Image
import numpy as np
import datetime
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from keras.utils import to_categorical

import pickle
import random

import matplotlib.pyplot as plt

from flickrapi import FlickrAPI
import pandas as pd
import sys

import csv
import requests
import time



class ezdata_images:

    def __init__(self):

        images = None
        labels = None
        synsets= None
        X      = None
        y      = None
        X_test = None
        y_test = None
        split  = None



    def import_classification(self,parameters):

        print ("EZ dataset : ", parameters["name"])
        images =[]
        labels =[]
        image_paths=[]
        synsets=[]

        print ('Loading :', parameters["path"])

        k=0
        tot=0
        for subdir in os.listdir(parameters["path"]):
            curdir = os.path.join(parameters["path"],subdir)
            i=0
            for filename in os.listdir(curdir):
                curimg = os.path.join(curdir, filename)
                img = Image.open(curimg)
                images.append(img)
                labels.append(k)
                image_paths.append(curimg)
                i=i+1
            synsets.append(subdir)
            k=k+1
            tot=tot+i
            print ('-- dir: ', subdir, '(',str(i),' images )')
        print ('Total images :', str(tot))
        self.images = images
        self.labels = labels
        self.image_paths = image_paths
        self.synsets = synsets

    def to_keras(self,resize=None):

        im=[]
        for image in self.images:
            r = image
            if resize is not None:
                r = image.resize((resize[0],resize[1]), Image.NEAREST)
            im.append(r)
        imgarray=list();
        for i in range(len(im)):
            tmp = np.array(im[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        if len(imgarray.shape)==1:
            print ("WARNING to_keras() : Image size heterogeneity !  Size of images into the dataset are not the same. You should try to use 'resize' parameters to make them homogenous.")
        if len(imgarray.shape)==3:
            imgarray = np.expand_dims(imgarray,axis=3)


        self.X = imgarray.astype('float32')
        self.y = np.asarray(self.labels)

        print ("Conversion to Keras format: Done")

    def gen_test(self,size=0.2,random_state=42):

        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=size,random_state=42)

        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        print ("Test set generation: Done")
        print ("Size : ", str(size))
        print ("Test set : ", self.X_test.shape[0], "images")

        return X_test, y_test


    def gen_train_val(self,size=0.2,random_state=42):

        X_train,X_valid,y_train,y_valid = train_test_split(self.X,self.y,test_size=size,random_state=42)

        print ("Train/Validation set generation: Done")
        print ("Size validation : ", str(size))
        print ("Training set : ", X_train.shape[0], "images")
        print ("Validation set     : ", X_valid.shape[0], "images")

        return X_train,y_train,X_valid,y_valid









    # def resize(self,resize0):
    #     for images in s









    def space():
        pass
