import os
import sys
import shutil
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

#圖片來源資料夾
picsrc = sys.argv[1]

autopick = sys.argv[2]

newpath = 'splitimage_bycos'
#相似度
comparethreshold = 0.999

def filename(dir):
    files = []
    for root, _, file in os.walk(dir):
        for f in file:
            absfile = os.path.join(root, f)
            if absfile.endswith(".png"):
                absfile = absfile.replace("\\", "/")
                if absfile not in files:
                    files.append(absfile)
    return files
    
class FilesCompare:
    def __init__(self):
        self.classes_sequence = []
    
    def compare(self, pic, file):
        splitpath = picsrc+"_"+newpath+'/0'
        splitpath = Path(splitpath)
        if not os.path.exists(splitpath):
            splitpath.mkdir(parents=True)
        maxsimple = 0
        j = 0
        for i in range(len(self.classes_sequence)):
            a = np.array(self.classes_sequence[i]).reshape(1, -1)
            b = np.array(pic).reshape(1, -1)
            result = cosine_similarity(a, b)
            if result > maxsimple:
                maxsimple = result
                j = i
        if maxsimple > comparethreshold:
            shutil.copy(file, splitpath)
        else:
            splitpath = picsrc+"_"+newpath+'/1'
            splitpath = Path(splitpath)
            if not os.path.exists(splitpath):
                splitpath.mkdir(parents=True)
            shutil.copy(file, splitpath)   

filecompare = FilesCompare()
    
for handpick in filename(picsrc):
    img = Image.open(handpick)
    img = ImageOps.grayscale(img)
    img1 = np.average(img, axis=1)
    filecompare.classes_sequence.append(img1)

for source in filename(autopick):
    img = Image.open(source)
    img = ImageOps.grayscale(img)
    img1 = np.average(img, axis=1)
    filecompare.compare(img1, source)
    
    