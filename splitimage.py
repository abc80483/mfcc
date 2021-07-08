import os
import sys
import shutil
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path

#圖片來源資料夾
picsrc = sys.argv[1]

newpath = 'splitimage'
#相似度
comparethreshold = 0.9

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
        self.paths = []
    
    def compare(self, pic, file):
        if not self.classes_sequence:
            self.classes_sequence.append(pic)
            for i in range(10000):
                splitpath = picsrc+"_"+newpath+'/'+str(i)
                splitpath = Path(splitpath)
                if not os.path.exists(splitpath):
                    splitpath.mkdir(parents=True)
                    break
            self.paths.append(splitpath)
            shutil.copy(file, splitpath)
        else:
            for i in range(len(self.classes_sequence)):
                a = self.classes_sequence[i].reshape(1, -1)
                b = pic.reshape(1, -1)
                result = cosine_similarity(a, b)
                
            if result > comparethreshold:
                shutil.copy(file, self.paths[i])
                
            else:
                self.classes_sequence.append(pic)
                print('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
                for i in range(10000):
                    splitpath = picsrc+"_"+newpath+'/'+str(i)
                    splitpath = Path(splitpath)
                    if not os.path.exists(splitpath):
                        splitpath.mkdir(parents=True)
                        break
                self.paths.append(splitpath)
                shutil.copy(file, splitpath)   

filecompare = FilesCompare()
    
for file in filename(picsrc):
    img = Image.open(file)
    img = ImageOps.grayscale(img)
    print(np.shape(img))
    img = np.average(img, axis=1)
    print(np.shape(img))
    filecompare.compare(img, file)
    
    
    