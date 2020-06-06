import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1.read data
fer2013_path = 'data/fer2013.csv'
data = pd.read_csv(fer2013_path)
data['pixels'] = data['pixels'].apply(lambda x: np.array([int(pixel) for pixel in x.split(' ')]))

#2.split data into sets
train_set   = data.loc[data['Usage'] == 'Training',['emotion','pixels']]
public_set  = data.loc[data['Usage'] == 'PublicTest',['emotion','pixels']]
private_set = data.loc[data['Usage'] == 'PrivateTest',['emotion','pixels']]

print(f"train_set :{train_set.shape[0]}")
print(f"public_set :{public_set.shape[0]}")
print(f"private_set:{private_set.shape[0]}")

# get sample image
image = train_set.loc[1, 'pixels']
label  = train_set.loc[1, 'emotion']
image = image.reshape(48,48)

plt.imshow(image, cmap= "gray", interpolation="nearest")    
plt.show()