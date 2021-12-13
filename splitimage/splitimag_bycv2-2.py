import os
import sys
import shutil
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import cv2

#圖片來源資料夾
picsrc = sys.argv[1]

autopick = sys.argv[2]

picsrc = picsrc.replace("\\", "/")
autopick = autopick.replace("\\", "/")

newpath = 'splitimage_bycv2'
#相似度
comparethreshold = 0.99999

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
            a = np.array(self.classes_sequence[i])
            b = np.array(pic)
            print("a{a}, b{b}".format(a = a.shape, b = b.shape))
            result = cv2.compareHist(a, b, 0)
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
    img = cv2.imread(handpick)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    filecompare.classes_sequence.append(H1)
    
for source in filename(autopick):
    img = cv2.imread(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    filecompare.compare(H1, source)
    
    
    