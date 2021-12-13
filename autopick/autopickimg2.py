import os
import sys
import shutil

#圖片來源資料夾(比較少)
picsrc = sys.argv[1]
#要挑選的圖片來源資料夾
soundsrc = sys.argv[2]
objdir = os.path.join(picsrc, "autopickimg")


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
    
if not os.path.exists(objdir):
    os.mkdir(objdir)
    
xs = filename(soundsrc)
    

for i in filename(picsrc):
    for x in xs:
        xfile = x.split('/')[-1]
        label, ifile = i.split('/')[-2:]
        if xfile == ifile:
            newpath = objdir+"/"+label
            #print(getpath)
            if not os.path.exists(newpath):
                os.mkdir(newpath)
            shutil.copy(x, newpath)

