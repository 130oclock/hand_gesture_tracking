# import necessary packages
import copy
import csv

import numpy as np
import cv2 as cv
import mediapipe as mp
import math

from landmarkInterpreter import LandmarkInterpreter
from motionInterpreter import MotionInterpreter
from handInput import Input

from collections import deque
from collections import Counter

# https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/0e737bb8c45ea03f6fafb1f5dbfe9246c34a8003

# initialize mediapipe hand tracking variables
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpStyles = mp.solutions.drawing_styles
mpDraw = mp.solutions.drawing_utils

# load the interpreter for hand gesture recognition
interpreter = LandmarkInterpreter()
tracking_set = set(interpreter.motionTrack)

# load the interpreter for motion gesture recognition
motion_interpreter = MotionInterpreter()

# create a list to store the history of the points
point_history_max_len = 15
point_history = deque(maxlen=point_history_max_len)
motion_history = deque(maxlen=point_history_max_len)

# important file paths
dataset_path = 'models/gesture_data.csv'
motion_dataset_path = 'models/motion_gesture_data.csv'
# set to true when collecting training data
saving_gesture_data = False
saving_motion_data = False

hand_input = Input()


def process_image(image):
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results


def draw_hands(image, results, key):
    # checking whether a hand is detected
    # hand_input.draw(image)
    if results.multi_hand_landmarks is not None:
        # loop through each hand
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # calculate a rectangle surrounding the hand
            bounding_rect = calc_bounding_rect(image, hand_landmarks)
            # convert landmarks to their coordinates in pixels
            landmarks = format_landmarks(image, hand_landmarks)
            # normalize the landmarks' coordinates between -1 and 1 to reduce variability
            normalized_landmarks = normalize_landmarks(landmarks)

            # predict what the hand gesture is
            class_id = interpreter(normalized_landmarks)
            class_name = interpreter.get_name(class_id)
            motion_name = ""

            # check if the gesture is tagged to track motion, if it is then add the index finger point to a list
            if class_id in tracking_set:
                point_history.append([landmarks[8], landmarks[0]])
            else:
                point_history.append([[0, 0], [0, 0]])

            if class_id == 3 and len(point_history) == point_history_max_len:
                hand_scale = math.dist(landmarks[5], landmarks[0]) / 10
                # normalize the motion of the hand
                normalized_motions = normalize_motion(point_history, hand_scale)
                # print(*normalized_motions)

                motion_id = motion_interpreter(normalized_motions)
                motion_history.append(motion_id)
                most_common_id = Counter(motion_history).most_common(1)[0][0]
                motion_name = motion_interpreter.get_name(most_common_id)

                if most_common_id == 1:  # flick up
                    hand_input.show()
                if most_common_id == 2:  # flick down
                    hand_input.hide()

            # when a key is pressed, log the current hand landmarks to the file
            key -= 48
            if 0 <= key <= 9:
                if saving_gesture_data is True:
                    log_data(key, normalized_landmarks, dataset_path)
                if saving_motion_data is True:
                    log_data(key, normalized_motions, motion_dataset_path)

            # draw green circles to display the point history
            draw_point_history(image)
            # draw a rectangle around the hand
            draw_bounding_rect(image, bounding_rect)
            # draw the hand's connections
            mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS,
                                  mpStyles.get_default_hand_landmarks_style())
            # draw text over the rectangle
            draw_text(image, bounding_rect, handedness, class_name + ' ' + motion_name)
        return image


# helper functions
def calc_bounding_rect(image, hand_landmarks):
    width, height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for index, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * width), width - 1)
        landmark_y = min(int(landmark.y * height), height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def format_landmarks(image, hand_landmarks):
    width, height = image.shape[1], image.shape[0]

    new_landmarks = np.empty((21, 2))
    for index, landmark in enumerate(hand_landmarks.landmark):  # loop through each landmark
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        new_landmarks[index] = [cx, cy]
    return new_landmarks


def normalize_landmarks(landmarks):
    # copy the landmarks
    temp_landmarks = copy.deepcopy(landmarks)

    base_x, base_y = 0, 0
    for index, temp_landmark in enumerate(temp_landmarks):
        if index == 0:
            base_x, base_y = temp_landmark[0], temp_landmark[1]
        temp_landmarks[index][0] = temp_landmarks[index][0] - base_x
        temp_landmarks[index][1] = temp_landmarks[index][1] - base_y

    # convert to 1D list
    temp_landmarks = temp_landmarks.flatten()

    # find the maximum value of all
    max_value = max(list(map(abs, temp_landmarks)))

    def normalize_(n):
        return n / max_value

    normalized_landmarks = list(map(normalize_, temp_landmarks))

    return normalized_landmarks


def normalize_motion(points, reference):
    distances = np.empty((point_history_max_len - 1, 2))
    for i in range(0, point_history_max_len - 2):
        point_i_x, point_i_y = points[i][0][0] - points[i][1][0], points[i][0][1] - points[i][1][1]
        point_f_x, point_f_y = points[i + 1][0][0] - points[i + 1][1][0], points[i + 1][0][1] - points[i + 1][1][1]
        distances[i] = [(point_f_x - point_i_x), (point_f_y - point_i_y)]
    distances = distances.flatten()

    max_value = reference  # max(list(map(abs, distances)))

    def normalize_(n):
        return n / max_value

    normalized_distances = list(map(normalize_, distances))
    return normalized_distances


def log_data(number, landmarks, file_path):
    with open(file_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmarks])
        f.close()


# debug drawing functions
def draw_bounding_rect(image, rect):
    cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), 1)


def draw_text(image, rect, handedness, details):
    text = handedness.classification[0].label[0:]
    cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[1] - 22 * 2),
                 (0, 0, 0), -1)
    cv.putText(image, text + ": ", (rect[0] + 5, rect[1] - 5 - 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image, details, (rect[0] + 5, rect[1] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


def draw_point_history(image):
    for point in point_history:
        cv.circle(image, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), 2)
