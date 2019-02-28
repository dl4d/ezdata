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

from keras.models import Model
from keras.layers import Input,Conv2D,Activation,MaxPooling2D,Flatten,Dense

#from keras.optimizers import Adam,SGD,RMSprop,Adamax,Adagrad,Adadelta,Nadam
from keras import optimizers

import pickle
import random

import matplotlib.pyplot as plt

from flickrapi import FlickrAPI
import pandas as pd
import sys

import csv
import requests
import time

from ezdata.utils import preprocess



class ezdata_images:

    def __init__(self):

        images = None
        labels = None
        synsets= None
        X      = None
        y      = None
        X_test = None
        y_test = None
        trainer = None
        type   = None



    def import_classification(self,parameters):

        print("\n")
        print (" * EZ dataset : ", parameters["name"]," * ")
        print("\n")
        images =[]
        labels =[]
        image_paths=[]
        synsets=[]
        self.type = "classification"

        print ('[X] Loading :', parameters["path"])

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
            print ('--- dir: ', subdir, '(',str(i),' images )')
        print ('--- Total images :', str(tot))
        self.images = images
        self.labels = labels
        self.image_paths = image_paths
        self.synsets = synsets
        print("\n")

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

        print ("[X] Conversion to Keras format: Done")
        print ("--- 'X' and 'y' tensors have been created.")
        print("\n")

    def gen_test(self,size=0.2,random_state=42):

        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=size,random_state=42)

        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        print ("[X] Test set generation (size = ,",str(size),",): Done")
        print ("--- Test set : ", self.X_test.shape[0], "images")
        print ("--- 'X_test' and 'y_test' tensors have been created.")
        print("\n")

        #return X_test, y_test


    def gen_train_val(self,size=0.2,random_state=42):

        X_train,X_valid,y_train,y_valid = train_test_split(self.X,self.y,test_size=size,random_state=42)

        print ("[X] Train/Validation set generation (size = ",str(size),"): Done")
        print ("--- Training set : ", X_train.shape[0], "images")
        print ("--- Validation set     : ", X_valid.shape[0], "images")
        print("\n")

        return X_train,y_train,X_valid,y_valid

    def assign(self,train=None,valid=None,scaler=None):
        ezt = ezdata_images_trainer()
        ezt.assign(train = (train[0],train[1]),valid = (valid[0],valid[1]),scaler = (scaler[0],scaler[1]))
        self.trainer= ezt


    def input_network(self):
        return Input(shape=self.trainer.X_train.shape[1:])

    def output_network(self,x):
        if self.type == "classification":
            if len(self.trainer.y_train.shape)==1:
                x = Dense(1) (x)
                x = Activation("linear") (x)
                return x
            else:
                x = Dense(self.trainer.y_train.shape[1]) (x)
                x = Activation("softmax") (x)
                return x

    def compile(self,network,optimizer="default",parameters=None):

        self.trainer.network = network

        #self.trainer.optimizer=optimizer

        if self.type == "classification":
            if len(self.trainer.y_train.shape)==1:
                loss = "mean_squarred_error"
                metrics = ["mse"]
            else:
                loss = "categorical_crossentropy"
                metrics = ["accuracy"]

        if optimizer == "default":
            optimizer = keras.optimizers.Adam()
            loss      = None
            metrics   = None
            self.trainer.network.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        else:
            self.trainer.network.compile(**optimizer)

    def train(self,parameters=None):
        #default parameters
        epochs = 10
        callbacks = None
        verbose = 1

        if parameters is not None:
            if "epochs" in parameters:
                epochs = parameters["epochs"]
            if "callbacks" in parameters:
                callbacks = parameters["callbacks"]
            if "verbose" in parameters:
                verbose = parameters["verbose"]

        history = self.trainer.network.fit(
                        self.trainer.X_train,
                        self.trainer.y_train,
                        validation_data=(self.trainer.X_valid,self.trainer.y_valid),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose = verbose
                        )

        print("\n")


    def evaluate(self):

        X_test = np.copy(self.X_test)
        y_test = np.copy(self.y_test)

        if hasattr(self.trainer,"scalerX"):
            if self.trainer.scalerX is not None:
                X_test,_  = preprocess(X_test,scaler=self.trainer.scalerX)

        if hasattr(self.trainer,"scalerY"):
            if self.trainer.scalerY is not None:
                if self.trainer.scalerY == "categorical":
                    y_test,_  = preprocess(y_test,type = "categorical")
                else:
                    y_test,_  = preprocess(y_test,scaler=self.trainer.scalery)

        print ("[X] Evaluation on Test set: ")
        p = self.trainer.network.evaluate(X_test,y_test,verbose=0)

        print ("--- Loss    : ", p[0])
        print ("--- Metrics : ", p[1])
        print("\n")

        return p

    def save(self,filename):

        print("[X] Save EZ as :", filename)
        if hasattr(self,"trainer"):
            if hasattr(self.trainer,"network"):
                network = self.trainer.network
                self.trainer.network=None
                network.save(filename+".h5")
                print("--- EZ trainer has been saved in :", filename,".h5")
            else:
                print("[Notice] No EZ trainer network to save has been found")
        else:
            print("[Notice] No EZ trainer to save has been found")


        filehandler = open(filename+".pkl","wb")
        pickle.dump(self,filehandler)
        print("--- EZ data has been saved in     :",filename,".pkl")
        print("\n")






class ezdata_images_trainer:

    def __init__(self):
        X_train   = None
        y_train   = None
        X_valid   = None
        y_valid   = None
        scalerX   = None
        scalerY   = None
        network   = None
        optimizer = None

    def assign(self,train = None, valid = None, scaler=None):
        self.X_train = train[0]
        self.y_train = train[1]
        self.X_valid = valid[0]
        self.y_valid = valid[1]
        if scaler is not None:
            self.scalerX = scaler[0]
            self.scalerY = scaler[1]





    def space():
        pass
