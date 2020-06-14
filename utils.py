
import numpy as np
import random
from keras.utils.np_utils import to_categorical

#3.make batch generator
#  using generator to save the memory
def sample_generator(df,batch_size):
    while True:
        index = np.arange(df.shape[0]) # get index of all picture
        random.shuffle(index)          # shuffle the index
        for i in range(0,int(np.ceil(df.shape[0]/batch_size))):
            id1 = i*batch_size
            id2 = min((i+1)*batch_size,len(index))
            batch_id = index[id1:id2]
            batch_X  = df.loc[batch_id,'pixels'].tolist()
            batch_Y  = df.loc[batch_id,'emotion'].tolist()
            #make the batch data to array format            
            batch_X  = [x.reshape(1,48,48,1) for x in batch_X]
            #regularize the data
            batch_X  = [x/128-1 for x in batch_X]
            batch_X  = np.concatenate(batch_X,axis=0)
            batch_Y  = to_categorical(batch_Y,num_classes=7)
            yield batch_X , batch_Y

def train_test_split(df,split):
    index = np.arange(df.shape[0]) # get index of all picture
    random.shuffle(index)
    split_id = round(df.shape[0]*split)
    train_index,test_index = index[:split_id],index[split_id:]
    train_df = df.loc[train_index].copy(deep=True).reset_index(drop=True)
    test_df  = df.loc[test_index].copy(deep=True).reset_index(drop=True)
    return train_df,test_df
    
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_labels():
    return {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',
                4:'Sad',5:'Surprise',6:'Neutral'}
