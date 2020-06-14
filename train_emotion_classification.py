import pandas as pd
import numpy as np
import random
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import matplotlib.pyplot as plt

from utils import train_test_split, sample_generator
from model import Emotion_classification_CNN
#1.read data
fer2013_path = 'data/fer2013.csv'
data = pd.read_csv(fer2013_path)
data['pixels'] = data['pixels'].apply(lambda x: np.array([int(pixel) for pixel in x.split(' ')]))

#2.split data into sets
train_set   = data.loc[data['Usage'] == 'Training',['emotion','pixels']].reset_index(drop=True)
public_set  = data.loc[data['Usage'] == 'PublicTest',['emotion','pixels']].reset_index(drop=True)
private_set = data.loc[data['Usage'] == 'PrivateTest',['emotion','pixels']].reset_index(drop=True)
# use train_set to train
# use public_set to evaluate

#5.train
batch_size = 128
# train_test_split
train_train,train_test = train_test_split(train_set,0.7)
model = Emotion_classification_CNN()
model.fit_generator(sample_generator(train_train,batch_size),
                    steps_per_epoch = np.ceil(train_train.shape[0]/batch_size),
                    validation_data = sample_generator(train_test,batch_size),
                    validation_steps= np.ceil(train_test.shape[0]/batch_size),
                    epochs = 10,verbose=2)
model.save('data/emotion_classification')
#6.evaluate
result = model.evaluate_generator(sample_generator(public_set,batch_size),
                         steps=np.ceil(public_set.shape[0]/batch_size))
print(f"public accuracy:{result[1]}")