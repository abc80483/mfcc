import os
import sys
import shutil

#圖片來源資料夾
picsrc = sys.argv[1]
#聲音來源資料夾
soundsrc = sys.argv[2]
objdir = "autopick"


def filename(dir):
    files = []
    for root, _, file in os.walk(dir):
        for f in file:
            absfile = os.path.join(root, f)
            if absfile.endswith(".png"):
                absfile = absfile.replace("\\", "/")
                absfile = absfile[:absfile.rfind("_")]
                if absfile not in files:
                    files.append(absfile)
    return files

for i in filename(picsrc):
    label, file = i.split('/')[-2:]
    label = label[:label.rfind('_')]
    labelfile = os.path.join(label, file)    
    getpath = os.path.join(soundsrc, labelfile)
    newpath = objdir+"/"+label
    print(getpath)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    if os.path.isfile(getpath):
        shutil.copy(getpath, newpath)
    else:
        print('error!')

