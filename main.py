import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from model import LetterDetector
from utils import extract_feature, toCSV, classes, num_classes

# Extract Feature for Training
# We will using SIBI datasets version V02
paths = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/training/"
csv_path = "/home/oleksandr/SignLanguage/hands_SIBI_training.csv"

if not os.path.exists(csv_path):
    print("The CSV file does not exist", csv_path, ",Going Create after Extraction")

    for dirlist in os.listdir(paths):
        for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
            print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".JPG"):
                    # print(os.path.join(root, filename), True)
                    (wristX, wristY, wristZ,
                     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                     thumb_McpX, thumb_McpY, thumb_McpZ,
                     thumb_IpX, thumb_IpY, thumb_IpZ,
                     thumb_TipX, thumb_TipY, thumb_TipZ,
                     index_McpX, index_McpY, index_McpZ,
                     index_PipX, index_PipY, index_PipZ,
                     index_DipX, index_DipY, index_DipZ,
                     index_TipX, index_TipY, index_TipZ,
                     middle_McpX, middle_McpY, middle_McpZ,
                     middle_PipX, middle_PipY, middle_PipZ,
                     middle_DipX, middle_DipY, middle_DipZ,
                     middle_TipX, middle_TipY, middle_TipZ,
                     ring_McpX, ring_McpY, ring_McpZ,
                     ring_PipX, ring_PipY, ring_PipZ,
                     ring_DipX, ring_DipY, ring_DipZ,
                     ring_TipX, ring_TipY, ring_TipZ,
                     pinky_McpX, pinky_McpY, pinky_McpZ,
                     pinky_PipX, pinky_PipY, pinky_PipZ,
                     pinky_DipX, pinky_DipY, pinky_DipZ,
                     pinky_TipX, pinky_TipY, pinky_TipZ,
                     annotated_image) = extract_feature(cv.imread(os.path.join(root, filename)))

                    if ((not wristX == 0) and (not wristY == 0)):
                        toCSV(csv_path, dirlist,
                              wristX, wristY, wristZ,
                              thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                              thumb_McpX, thumb_McpY, thumb_McpZ,
                              thumb_IpX, thumb_IpY, thumb_IpZ,
                              thumb_TipX, thumb_TipY, thumb_TipZ,
                              index_McpX, index_McpY, index_McpZ,
                              index_PipX, index_PipY, index_PipZ,
                              index_DipX, index_DipY, index_DipZ,
                              index_TipX, index_TipY, index_TipZ,
                              middle_McpX, middle_McpY, middle_McpZ,
                              middle_PipX, middle_PipY, middle_PipZ,
                              middle_DipX, middle_DipY, middle_DipZ,
                              middle_TipX, middle_TipY, middle_TipZ,
                              ring_McpX, ring_McpY, ring_McpZ,
                              ring_PipX, ring_PipY, ring_PipZ,
                              ring_DipX, ring_DipY, ring_DipZ,
                              ring_TipX, ring_TipY, ring_TipZ,
                              pinky_McpX, pinky_McpY, pinky_McpZ,
                              pinky_PipX, pinky_PipY, pinky_PipZ,
                              pinky_DipX, pinky_DipY, pinky_DipZ,
                              pinky_TipX, pinky_TipY, pinky_TipZ, )

                    else:
                        print(os.path.join(root, filename), "Hand does not have landmarks")

    print("===================Feature Extraction for TRAINING is Completed===================")

# Extract Feature for Validation
# We will using SIBI datasets version V02
paths = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/validation/"
csv_path = "/home/oleksandr/SignLanguage/hands_SIBI_training.csv"

if not os.path.exists(csv_path):
    print("The CSV file does not exist", csv_path, ",Going Create after Extraction")

    for dirlist in os.listdir(paths):
        for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
            print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".JPG"):
                    # print(os.path.join(root, filename), True)
                    (wristX, wristY, wristZ,
                     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                     thumb_McpX, thumb_McpY, thumb_McpZ,
                     thumb_IpX, thumb_IpY, thumb_IpZ,
                     thumb_TipX, thumb_TipY, thumb_TipZ,
                     index_McpX, index_McpY, index_McpZ,
                     index_PipX, index_PipY, index_PipZ,
                     index_DipX, index_DipY, index_DipZ,
                     index_TipX, index_TipY, index_TipZ,
                     middle_McpX, middle_McpY, middle_McpZ,
                     middle_PipX, middle_PipY, middle_PipZ,
                     middle_DipX, middle_DipY, middle_DipZ,
                     middle_TipX, middle_TipY, middle_TipZ,
                     ring_McpX, ring_McpY, ring_McpZ,
                     ring_PipX, ring_PipY, ring_PipZ,
                     ring_DipX, ring_DipY, ring_DipZ,
                     ring_TipX, ring_TipY, ring_TipZ,
                     pinky_McpX, pinky_McpY, pinky_McpZ,
                     pinky_PipX, pinky_PipY, pinky_PipZ,
                     pinky_DipX, pinky_DipY, pinky_DipZ,
                     pinky_TipX, pinky_TipY, pinky_TipZ,
                     annotated_image) = extract_feature(cv.imread(os.path.join(root, filename)))

                    if ((not wristX == 0) and (not wristY == 0)):
                        toCSV(csv_path, dirlist,
                              wristX, wristY, wristZ,
                              thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                              thumb_McpX, thumb_McpY, thumb_McpZ,
                              thumb_IpX, thumb_IpY, thumb_IpZ,
                              thumb_TipX, thumb_TipY, thumb_TipZ,
                              index_McpX, index_McpY, index_McpZ,
                              index_PipX, index_PipY, index_PipZ,
                              index_DipX, index_DipY, index_DipZ,
                              index_TipX, index_TipY, index_TipZ,
                              middle_McpX, middle_McpY, middle_McpZ,
                              middle_PipX, middle_PipY, middle_PipZ,
                              middle_DipX, middle_DipY, middle_DipZ,
                              middle_TipX, middle_TipY, middle_TipZ,
                              ring_McpX, ring_McpY, ring_McpZ,
                              ring_PipX, ring_PipY, ring_PipZ,
                              ring_DipX, ring_DipY, ring_DipZ,
                              ring_TipX, ring_TipY, ring_TipZ,
                              pinky_McpX, pinky_McpY, pinky_McpZ,
                              pinky_PipX, pinky_PipY, pinky_PipZ,
                              pinky_DipX, pinky_DipY, pinky_DipZ,
                              pinky_TipX, pinky_TipY, pinky_TipZ, )

                    else:
                        print(os.path.join(root, filename), "Hand does not have landmarks")

    print("===================Feature Extraction for VALIDATION is Completed===================")

# Read CSV file for Training the model using Pandas
df_train = pd.read_csv("hands_SIBI_training.csv", header=0)

# First we must sort the values of the dataset according to the Alphabets
df_train = df_train.sort_values(by=["class_type"])

# Read CSV file for Validation or Testing the Model using Pandas
df_test = pd.read_csv("hands_SIBI_validation.csv", header=0)

# First we must sort the values of the dataset according to the Alphabets
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
# Check Array Shape before transformation
print(x_train.shape)
print(x_test.shape)

# Since the array shape is 1x1, we must turn it into 1x10x1 so we can feed it into the model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Check Array Shape after transformation
print(x_train.shape)
print(x_test.shape)

# Number of classes according standard Indonesian Language Alphabets


# Using the Keras.Utils to put the label categorically
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# One Dimensional Convolutional Neural Network model, Train will be feed to 1 Dimension Convolutional Neural Network

# Directly from Imageset Dataset Testing
# Load Image and do Feature Extraction
path_to_image = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/test/F (3).jpg"

model = LetterDetector(x_train.shape[1:3])
# model.train(x_train, y_train, x_test, y_test, epochs=100, batch_size=32)
# model.summary()
# model.save("model.h5")
model.load("model.h5")

input_IMG = cv.imread(path_to_image)

# Print the Prediction
print(model.predict(input_IMG))
print(model.predict_classes(input_IMG))

# Print prediction using defined Classes
predictions = model.predict_classes(input_IMG)
for alphabets, values in classes.items():
    if values == predictions:
        print("Possible Alphabet according to the input : ", alphabets)
