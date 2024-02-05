import numpy as np
import mediapipe as mp
import os
import csv
import cv2
import pandas as pd
from keras.src.utils import to_categorical

from data.paths import validation_paths, validation_csv_path

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

num_classes = 26
classes = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}


def extract_feature(image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
        image_height, image_width, _ = image.shape

        if not results.multi_hand_landmarks:
            return get_default_landmarks(image)

        annotated_image = cv2.flip(image.copy(), 1)
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = get_hand_landmarks(mp_hands, hand_landmarks, image_width, image_height)
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return landmarks, annotated_image


def get_default_landmarks(image):
    # Set all landmarks to zero
    landmarks = [0] * 63
    annotated_image = 0
    return landmarks, annotated_image


def get_hand_landmarks(mp_hands, hand_landmarks, image_width, image_height):
    landmarks = []
    for landmark_type in mp_hands.HandLandmark:
        landmark = hand_landmarks.landmark[landmark_type]
        landmarks.extend([landmark.x * image_width, landmark.y * image_height, landmark.z])
    return landmarks


def to_csv(filecsv, class_type, *landmarks):
    coords = ['X', 'Y', 'Z']
    header = ["class_type"]
    header.extend([f"{part}_{coord}" for part in mp.solutions.hands.HandLandmark._member_names_ for coord in coords])
    if os.path.isfile(filecsv):
        append_to_csv(filecsv, header, class_type, *landmarks)
    else:
        create_new_csv(filecsv, header, class_type, *landmarks)


def append_to_csv(filecsv, header, class_type, *data):
    with open(filecsv, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_type, *data])


def create_new_csv(filecsv, header, class_type, *data):
    with open(filecsv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow([class_type, *data])


def process_data(paths, csv_path):
    if not os.path.exists(csv_path):
        print("The CSV file does not exist:", csv_path, ", going to create it after extraction")

        for dirlist in os.listdir(paths):
            for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
                print("Inside Folder", dirlist, "Consisting of:", len(filenames), "Imageset")
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg")):
                        image_path = os.path.join(root, filename)
                        features = extract_feature(cv2.imread(image_path))

                        if features and features[0] != 0:  # Assuming wristX is the first element of features
                            to_csv(csv_path, dirlist, *features)
                        else:
                            print(image_path, "Hand does not have landmarks")

        print("===================Feature Extraction is Completed===================")


def preparation_data():
    process_data(validation_paths, validation_csv_path)
    # Read CSV file for Training the model using Pandas
    df_train = pd.read_csv("data/hands_SIBI_training.csv", header=0)
    df_train = df_train.sort_values(by=["class_type"])

    # Read CSV file for Validation or Testing the Model using Pandas
    df_test = pd.read_csv("data/hands_SIBI_validation.csv", header=0)
    df_test = df_test.sort_values(by=["class_type"])

    # Put Categorical using Pandas
    df_train["class_type"] = pd.Categorical(df_train["class_type"])
    df_train["class_type"] = df_train.class_type.cat.codes

    df_test["class_type"] = pd.Categorical(df_test["class_type"])
    df_test["class_type"] = df_test.class_type.cat.codes

    # Copy Label and Feature for training
    y_train = df_train.pop("class_type")
    x_train = df_train.copy()

    y_test = df_test.pop("class_type")
    x_test = df_test.copy()

    # Copied Features turn to Array by using NumPy
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Since the array shape is 1x1, we must turn it into 1x10x1 so we can feed it into the model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Using the Keras.Utils to put the label categorically
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test