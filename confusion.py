from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
model = load_model(r'C:\Users\user\Documents/model.h5')
files = sys.argv[1:]

IMAGE_SIZE = (224, 224)
alabels = ["Acridotherestristis_mfcc",
            "Aegithinatiphia_mfcc",
            "Amaurornisphoenicurus_mfcc",
            "Brantacanadensis_mfcc",
            "Cyanistescaeruleus_mfcc",
            "Erithacusrubecula_mfcc",
            "Ficedulaparva_mfcc",
            "Hirundorustica_mfcc",
            "Juncohyemalis_mfcc",
            "Pycnonotussinensis_mfcc"]
            
alabels15 = ["Acridotherestristis_mfcc",
            "Aegithinatiphia_mfcc",
            "Alaudagulgula_mfcc",
            "Alcedoatthis_mfcc",
            "Amaurornisphoenicurus_mfcc",
            "Brantacanadensis_mfcc",
            "Cyanistescaeruleus_mfcc",
            "Erithacusrubecula_mfcc",
            "Ficedulaparva_mfcc",
            "Hirundorustica_mfcc",
            "Juncohyemalis_mfcc",
            "Pycnonotussinensis_mfcc",
            "Sittasomusgriseicapillus_mfcc",
            "Streptopeliadecaocto_mfcc",
            "Turdusviscivorus_mfcc"]
            
alabels20 = ["Acridotherestristis_mfcc",
            "Aegithinatiphia_mfcc",
            "Alaudagulgula_mfcc",
            "Alcedoatthis_mfcc",
            "Amaurornisphoenicurus_mfcc",
            "Ardeaalba_mfcc",
            "Brantacanadensis_mfcc",
            "Cyanistescaeruleus_mfcc",
            "Dendrocoposleucotos_mfcc",
            "Erithacusrubecula_mfcc",
            "Ficedulaparva_mfcc",
            "Hirundorustica_mfcc",
            "Horornisfortipes_mfcc",
            "Juncohyemalis_mfcc",
            "Laniuscollurio_mfcc",
            "Periparusater_mfcc",
            "Pycnonotussinensis_mfcc",
            "Sittasomusgriseicapillus_mfcc",
            "Streptopeliadecaocto_mfcc",
            "Turdusviscivorus_mfcc"]
            
labels = np.arange(0, 10)
labels15 = np.arange(0, 15)
labels20 = np.arange(0, 20)
test_datagen = ImageDataGenerator(fill_mode='wrap')
test_batches = test_datagen.flow_from_directory(files[0],
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=1)







predictions = model.predict_generator(test_batches)

predictions = np.argmax(predictions, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(test_batches.classes, predictions, labels=labels20)

ax = sns.heatmap(cm, annot=True, vmax=100, yticklabels=labels20, xticklabels=labels20)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.xlabel("predict classes")
plt.ylabel("True classes")
plt.savefig("confusion_matrix.png")
plt.show()
print(cm)
print(classification_report(test_batches.classes, predictions, target_names=alabels20))