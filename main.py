import pandas as pd
import numpy as np
import random
import warnings
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import matplotlib.pyplot as plt

#1.read data
path = "D:/python/contest/picture/fer2013/"
file = "fer2013.csv"
data = pd.read_csv(path+file)
data['pixels'] = data['pixels'].apply(lambda x: np.array([int(pixel) for pixel in x.split(' ')]))

#2.split data into sets
train_set   = data.loc[data['Usage'] == 'Training',['emotion','pixels']].reset_index(drop=True)
public_set  = data.loc[data['Usage'] == 'PublicTest',['emotion','pixels']].reset_index(drop=True)
private_set = data.loc[data['Usage'] == 'PrivateTest',['emotion','pixels']].reset_index(drop=True)
# use train_set to train
# use public_set to evaluate

