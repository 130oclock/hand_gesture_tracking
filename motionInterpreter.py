import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model


class MotionInterpreter:
    def __init__(self,
                 model_path='models/motion_gesture_model',
                 name_path='models/motion.names',
                 number_of_threads=1,
                 min_conf=0.3,
                 invalid_value=-1
                 ):
        # read and store a list of the gesture names
        with open(name_path, 'r') as names:
            self.classnames = names.read().split('\n')
            names.close()
        self.min_conf = min_conf
        self.invalid_value = invalid_value

        # load the model and set up the interpreter
        converter = tf.lite.TFLiteConverter.from_keras_model(load_model(model_path))
        tflite_model = converter.convert()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=number_of_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]['index']
        self.output = self.interpreter.get_output_details()[0]['index']

    # takes in a list of points and returns the index of the motion it is most confident
    def __call__(self,
                 point_history
                 ):
        self.interpreter.set_tensor(self.input, np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output)
        index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[index] < self.min_conf:
            index = self.invalid_value

        return index

    # takes in the motion index and returns its name
    def get_name(self,
                 index):
        if index == -1:
            return "Invalid"
        return self.classnames[index]
