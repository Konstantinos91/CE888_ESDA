from timeit import default_timer as timer
import matplotlib.pyplot as plt

import cPickle as pkl
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

import sys
sys.path.insert(0, 'infoGA_ssamot')
from keras_helper import NNWeightHelper
from snes import SNES

from tensorflow.examples.tutorials.mnist import input_data

# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 30
# how many times we will loop over ask()/tell()
GENERATIONS = 40


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


# ----------------- Load MNIST & MNIST-M datasets -----------------#
# -------- MNIST-M
mnistm = pkl.load(open('tf-dann_pumpikano/mnistm_data.pkl', 'rb'))
mnistm_train_data = mnistm['train']
mnistm_eval_data = mnistm['test']
mnistm_valid_data = mnistm['valid']

# -------- MNIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

print(len(mnist_train))

# Create a mixed dataset
num_test = 10000
combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_eval_data[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        np.tile([0., 1.], [num_test, 1])])

# the data, split between train and test sets
train_num = 50000
test_num = 10000
(x_train, y_train) = mnist_train[:train_num], mnist.train.labels[:train_num]
#(x_test, y_test) = combined_test_imgs[:test_num], combined_test_labels[:test_num]
(x_test, y_test) = mnistm_eval_data[:test_num], mnist.test.labels[:test_num]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(len(x_train))
print (len(x_test))

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
                 input_shape=(28, 28, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(6, activation='relu'))
model.summary()
'''

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
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
        subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)
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
    plt.title("Domain Adaptation -- MNIST-M")
    plt.xlabel("Generation", fontsize=14, color='blue')
    plt.ylabel("Loss", fontsize=14, color='blue')
    plt.savefig("mnistm_loss.jpg")
    plt.show()

if __name__ == '__main__':
    main()