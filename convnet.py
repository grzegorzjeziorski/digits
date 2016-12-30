import sys
import pandas as pd

from keras.layers.core import Flatten
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D
from keras.models import Sequential
from argparse import ArgumentParser

from keras.utils import np_utils

def load_data_frame(path):
    return pd.read_csv(path)

def model0():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model1():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model2():
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model3():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_args_parser():
    parser = ArgumentParser(description="Data generator")
    parser.add_argument('--model', help='Model to test', type=int)
    parser.add_argument('--train', help='Path of the train file')
    parser.add_argument('--test', help='Path of the test file')
    return parser

def cli(cli_args):
    parser = build_args_parser()
    args = parser.parse_args(cli_args)
    if args.train and args.test:
        models = [model0(), model1(), model2(), model3()]
        model = models[args.model]
        train_df = pd.read_csv(args.train)
        test_df = pd.read_csv(args.test)
        train_matrices = train_df.as_matrix()[:, 1:]
        train_labels = train_df.as_matrix()[:, 0].astype('int')
        test_matrices = test_df.as_matrix()[:, 1:]
        test_labels = test_df.as_matrix()[:, 0].astype('int')
        train_matrices = train_matrices.reshape(train_matrices.shape[0], 1, 28, 28).astype('float32')
        test_matrices = test_matrices.reshape(test_matrices.shape[0], 1, 28, 28).astype('float32')
        train_matrices = train_matrices / 255.0
        test_matrices = test_matrices / 255.0
        train_labels = np_utils.to_categorical(train_labels)
        test_labels = np_utils.to_categorical(test_labels)
        model.fit(train_matrices, train_labels, validation_data=(test_matrices, test_labels), nb_epoch=10, batch_size=200, verbose=100)
        scores = model.evaluate(test_matrices, test_labels, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    else:
        parser.print_usage()

cli(sys.argv[1:])

