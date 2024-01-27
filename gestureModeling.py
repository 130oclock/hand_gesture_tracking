
import numpy as np
import tensorflow as tf
from tensorflow.python import keras

# initialize hand related variables
hand_dataset_path = 'models/gesture_data.csv'
hand_model_path = 'models/keypoint_model'

number_of_classes = 7
number_of_key_points = 21


def train_hand_model():
    xs = np.loadtxt(hand_dataset_path, dtype='float32', delimiter=',',
                    usecols=list(range(1, (number_of_key_points * 2) + 1)))
    ys = np.loadtxt(hand_dataset_path, dtype='int32', delimiter=',', usecols=0)

    model = keras.Sequential([
        keras.layers.InputLayer((number_of_key_points * 2, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(number_of_classes, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(xs, ys, epochs=1000, batch_size=128)

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

    model.save(hand_model_path)
    # tflite_model.save(tflite_model_path


# initialize motion related variables
motion_dataset_path = 'models/motion_gesture_data.csv'
motion_model_path = 'models/motion_gesture_model'

number_of_motions = 5
number_of_motion_key_points = 14


def train_motion_model():
    xs = np.loadtxt(motion_dataset_path, dtype='float32', delimiter=',',
                    usecols=list(range(1, (number_of_motion_key_points * 2) + 1)))
    ys = np.loadtxt(motion_dataset_path, dtype='int32', delimiter=',', usecols=0)

    model = keras.Sequential([
        keras.layers.InputLayer((number_of_motion_key_points * 2,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(number_of_motions, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(xs, ys, epochs=1000, batch_size=128)

    model.save(motion_model_path)


if __name__ == '__main__':
    # train_hand_model()
    train_motion_model()
