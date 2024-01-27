import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model


class LandmarkInterpreter:
    def __init__(self,
                 model_path='models/keypoint_model',
                 name_path='models/gesture.names',
                 number_of_threads=1
                 ):
        # read and store a list of the gesture names
        with open(name_path, 'r') as names:
            self.classnames = names.read().split('\n')
            names.close()

        self.motionTrack = []
        for i, line in enumerate(self.classnames):
            data = line.split('#')
            self.classnames[i] = data[0]
            if len(data) > 1:
                if data[1] == 'm':
                    self.motionTrack.append(i)

        # load the model and set up the interpreter
        converter = tf.lite.TFLiteConverter.from_keras_model(load_model(model_path))
        tflite_model = converter.convert()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=number_of_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]['index']
        self.output = self.interpreter.get_output_details()[0]['index']

    # takes in a list of landmarks and returns the index of the gesture it is most confident
    def __call__(self,
                 landmarks
                 ):
        self.interpreter.set_tensor(self.input, np.array([landmarks], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output)
        index = np.argmax(np.squeeze(result))
        return index

    # takes in the gesture index and returns its name
    def get_name(self,
                 index):
        return self.classnames[index]
