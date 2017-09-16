"""Testing K-means Optimizer.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from keras.models import Model
from keras.layers import Conv2D, Input

from convk.kmeans import KMeansOptimizer

# define model
img_input = Input(shape=(32, 32, 3))
x = Conv2D(16, (7, 7), padding="same", name="kmeans_1")(img_input)
x = Conv2D(16, (7, 7), padding="same", name="kmeans_2")(x)

model = Model(img_input, x)

optimizer = KMeansOptimizer(model)

print (optimizer.layer_idx)
