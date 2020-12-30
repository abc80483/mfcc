import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
_, (test_x, test_y) = mnist.load_data()

model = load_model("test.h5")

test_x = test_x.reshape(-1, 28, 28, 1)
test_x = test_x / 255

test_y = np_utils.to_categorical(test_y, num_classes=10)

print(test_x.shape[0])

score = model.evaluate(test_x,test_y)
print(score)
