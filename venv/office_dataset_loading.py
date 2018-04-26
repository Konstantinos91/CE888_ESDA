from timeit import default_timer as timer
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, 'infoGA_ssamot')
from keras_helper import NNWeightHelper
from snes import SNES

from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2
import os
from imutils import paths

# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

# ----------------- Random Forest Classifier -----------------#

def train_classifier(model, X, y):
    X_features = model.predict(X)
    clf = RandomForestClassifier()

    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred


def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)


# ----------------- Load Office-31 dataset -----------------#

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 31

# -------- Amazon Images --------#
amazonPath = '../data/domain_adaptation_images/amazon/images/'

amazonData = []
amazonLabels = []

amazonPath = sorted(list(paths.list_files(amazonPath)))

for imagePath_a in amazonPath:
    img_a = cv2.imread(imagePath_a)
    img_a = cv2.resize(img_a, (28, 28))
    img_a = img_to_array(img_a)
    amazonData.append(img_a)

    label_a = imagePath_a.split(os.path.sep)[-2]
    amazonLabels.append(label_a)

amazonData = np.array(amazonData, dtype="float32") / 255.0
amazonLabels = np.array(amazonLabels)

# -------- DSLR Images --------#
dslrPath = '../data/domain_adaptation_images/dslr/images/'

dslrData = []
dslrLabels = []

dslrPath = sorted(list(paths.list_files(dslrPath)))

for imagePath_d in dslrPath:
    img_d = cv2.imread(imagePath_d)
    img_d = cv2.resize(img_d, (28, 28))
    img_d = img_to_array(img_d)
    dslrData.append(img_d)

    label_d = imagePath_d.split(os.path.sep)[-2]
    dslrLabels.append(label_d)

dslrData = np.array(dslrData, dtype="float32") / 255.0
dslrLabels = np.array(dslrLabels)

# -------- Webcam Images --------#
webcamPath = '../data/domain_adaptation_images/webcam/images/'

webcamData = []
webcamLabels = []

webcamPath = sorted(list(paths.list_files(webcamPath)))

for imagePath_w in webcamPath:
    img_w =cv2.imread(imagePath_w)
    img_w = cv2.resize(img_w, (28, 28))
    img_w = img_to_array(img_w)
    webcamData.append(img_w)

    label_w = imagePath_w.split(os.path.sep)[-2]
    webcamLabels.append(label_w)

webcamData = np.array(webcamData, dtype="float32") / 255.0
webcamLabels = np.array(webcamLabels)

'''
datagen = ImageDataGenerator()
amazon = datagen.flow_from_directory(amazonPath, target_size= (256, 256), batch_size=2)
dslr = datagen.flow_from_directory(dslrPath, target_size= (256, 256), batch_size=2)
webcam = datagen.flow_from_directory(webcamPath, target_size= (256, 256), batch_size=2)
'''

# the data, split between train and test sets
#(x_train, y_train) = amazonData, amazonLabels
(x_train, y_train) = dslrData, dslrLabels
#(x_test, y_test) = dslrData, dslrLabels
(x_test, y_test) = webcamData, webcamLabels

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28, 28, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.summary()

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(6, activation='relu'))
'''

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
print("compilation is over")

nnw = NNWeightHelper(model)
weights = nnw.get_weights()

def main():
    loss = []
    print("Total number of weights to evolve is:", weights.shape)

    all_examples_indices = list(range(x_train.shape[0]))

    clf, _ = train_classifier(model, x_train, y_train)

    y_pred = predict_classifier(model, clf, x_test)
    print(y_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(y_test, y_pred)

    print('Non-trained NN Test accuracy:', test_accuracy)
    # print('Test MSE:', test_mse)

    snes = SNES(weights, 1, POPULATION_SIZE)
    for i in range(0, GENERATIONS):
        start = timer()
        asked = snes.ask()

        # to be provided back to snes
        told = []
        # use a small number of training samples for speed purposes
        subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=True)
        # evaluate on another subset
        subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)

        # iterate over the population
        for asked_j in asked:
            # set nn weights
            nnw.set_weights(asked_j)
            # train the classifer and get back the predictions on the training data
            clf, _ = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])

            # calculate the predictions on a different set
            y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid])
            score = accuracy_score(y_train[subsample_indices_valid], y_pred)

            # clf, _ = train_classifier(model, x_train, y_train)
            # y_pred = predict_classifier(model, clf, x_test)
            # score = accuracy_score(y_test, y_pred)
            # append to array of values that are to be returned
            told.append(score)

        l = snes.tell(asked, told)
        loss.append(l)
        end = timer()
        print("It took", end - start, "seconds to complete generation", i + 1)

    nnw.set_weights(snes.center)

    clf, _ = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test)

    print(y_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(y_test, y_pred)

    print('Test accuracy:', test_accuracy)

    plt.plot(loss)
    plt.title("Domain Adaptation -- Office-31")
    plt.xlabel("Generation", fontsize=14, color='blue')
    plt.ylabel("Loss", fontsize=14, color='blue')
    plt.savefig("office31_loss_dw.jpg")
    plt.show()

if __name__ == '__main__':
    main()