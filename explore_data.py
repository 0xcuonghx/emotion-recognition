import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fer2013_path = 'data/fer2013.csv'
raw_data = pd.read_csv(fer2013_path)

# raw_data.info()
# raw_data.head()
# raw_data["Usage"].value_counts()

# See pricture
img = raw_data["pixels"][0]
value = img.split(" ")
x_pixels = np.array(value, 'float32')
x_pixels /= 255
x_reshaped = x_pixels.reshape(48, 48)

plt.imshow(x_reshaped, cmap= "gray", interpolation="nearest")    
plt.axis("off")
plt.show()