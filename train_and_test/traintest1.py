from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np
from tensorflow.python import keras
from keras.utils import plot_model
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# 從參數讀取圖檔路徑
files = sys.argv[1:]


# 載入訓練好的模型
net = load_model(r'C:\Users\user\Documents/model.h5')
IMAGE_SIZE = (224, 224)

test_datagen = ImageDataGenerator(fill_mode='wrap')
test_batches = test_datagen.flow_from_directory(files[0],
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=1)

def filename(dir):
    #使用一個list來存所有wav檔的檔名
    files = []
    for root, _, file in os.walk(dir):
        for f in file:
            absfile = os.path.join(root, f)
            if absfile.endswith(".png"):
                absfile = absfile.replace("\\", "/")
                files.append(absfile)
    return files

"""
for f in filename(files[0]):
    img = image.load_img(f, target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    np.append(testlist, x)
"""
    



loss, acc = net.evaluate(test_batches)
print('loss:',loss)
print('acc', acc)

"""
# 辨識每一張圖
for f in filname(files):
    img = image.load_img(f, target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.evaluate(x, label)[0]
    top_inds = pred.argsort()[::-1][:]
    print(np.argmax(pred))
    print(f)
    '''for i in top_inds:
        print('    {:.3f} '.format(pred[i]))'''
    for i in range(10):
        print('{:.3f}'.format(pred[i]))
plot_model(net, to_file='model.png')"""