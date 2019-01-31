from matplotlib.pyplot import imread
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import resnet50

filepath = r"E:\Users\Tejan\PycharmProjects\FacialBeauty\SCUT-FBP5500_v2\\"

def import_images_and_scores(filepath):
    f = open(filepath+r"train_test_files\60_40_split\train.txt",'r')

    images = []
    scores = []
    for line in f:
        line = line.replace('\n','')
        line = line.split(' ')
        image = imread(filepath+r"Images\\"+line[0])
        images.append(image)
        scores.append(line[1])
    return images,scores

#images,scores = import_images_and_scores(filepath)
rn = resnet50.ResNet50()
model = Sequential()
model.add(rn)
model.add(Dense(1))

model.layers[0].trainable = False

print(model.summary())

model.compile(loss='mse',optimizer=Adam())

#model.fit(batch_size=32,x=images,y=scores, epochs=15)


#model.save_weights('FBP.h5')
