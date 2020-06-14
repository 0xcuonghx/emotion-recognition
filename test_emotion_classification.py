
from utils import train_test_split, sample_generator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

batch_size = 128
fer2013_path = 'data/fer2013.csv'
data = pd.read_csv(fer2013_path)
data['pixels'] = data['pixels'].apply(lambda x: np.array([int(pixel) for pixel in x.split(' ')]))
public_set  = data.loc[data['Usage'] == 'PublicTest',['emotion','pixels']].reset_index(drop=True)
emotion_classifier=load_model('./model/emotion_classification', compile=False)
opt = Adam(lr=0.0001, decay=10e-6)
emotion_classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
result = emotion_classifier.evaluate_generator(sample_generator(public_set,batch_size),
                         steps=np.ceil(public_set.shape[0]/batch_size))
print(f"accuracy:{result[1]}")