from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
import pickle
import os

def ez_load(filename):

    if os.path.isfile(filename + ".pkl"):
        print("[X] Loading EZ data    : ", filename+".pkl")
        filehandler = open(filename+".pkl", 'rb')
        ez = pickle.load(filehandler)
    else:
        print("[Fail] Loading EZ data    : ", filename+".pkl doesn't exist !")

    if os.path.isfile(filename + ".h5"):
        print("[X] Loading EZ trainer : ", filename+".h5")
        ez.trainer.network = load_model(filename+".h5")
    else:
        print("[X] No EZ trainer to load.")
    return ez






def preprocess(data,type=None,scaler=None):
    if type=="minmax":
        return minmax_scaling(data)
    if type=="categorical":
        return categorical_transform(data)
    if type==None:
        if scaler=="None":
            print ("[Fail: preprocess()] No 'type' nor 'scaler' defined")
            return
        else:
            return scaler_scaling(data,scaler)


def minmax_scaling(data):
    scalers=[]
    for i in range(data.shape[3]):
        scaler = MinMaxScaler()
        shape_before = data[:,:,:,i].shape
        a = data[:,:,:,i].reshape(-1,1)
        scalers.append(scaler.fit(a))
        b = scalers[i].transform(a)
        data[:,:,:,i] = b.reshape(shape_before)
    #print ("[X] Preprocessing : MinMax")
    #print("\n")
    return data,scalers

def standard_scaling(data):

    scalers=[]
    for i in range(data.shape[3]):
        scaler = StandardScaler()
        shape_before = data[:,:,:,i].shape
        a = data[:,:,:,i].reshape(-1,1)
        scalers.append(scaler.fit(a))
        b = scalers[i].transform(a)
        data[:,:,:,i] = b.reshape(shape_before)
    #print ("[X] Preprocessing : Standard")
    #print("\n")
    return data,scalers

def categorical_transform(data):
    #print ("[X] Preprocessing : Categorical")
    #print("\n")
    return to_categorical(data),"categorical"

def scaler_scaling(data,scaler):
    for i in range(len(scaler)):
        shape_before = data[:,:,:,i].shape
        a = data[:,:,:,i].reshape(-1,1)
        b = scaler[i].transform(a)
        data[:,:,:,i] = b.reshape(shape_before)
    #print ("[X] Preprocessing using scalers.")
    #print("\n")
    return data,scaler


def gen_trainval(X,y,size=0.2,random_state=42):
    X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=size,random_state=42)
    print ("[X] Train/Validation set generation (size = ,",str(size),",): Done")
    print ("--- Train set      : ", X_train.shape[0], "images")
    print ("--- Validation set : ", X_valid.shape[0], "images")
    print("\n")

    return X_train,X_valid,y_train,y_valid
