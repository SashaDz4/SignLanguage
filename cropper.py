import os

import cv2
import numpy as np

# classes for ukrainian letters without "Й", "Ї", "Щ", "Ґ"
classes = {"А": 0, "Б": 1, "В": 2, "Г": 3, "Д": 4, "Е": 5, "Є": 6, "Ж": 7, "З": 8, "И": 9, "І": 10, "К": 11, "Л": 12,
           "М": 13, "Н": 14, "О": 15, "П": 16, "Р": 17, "С": 18, "Т": 19, "У": 20, "Ф": 21, "Х": 22, "Ц": 23, "Ч": 24,
           "Ш": 25, "Ь": 26, "Ю": 27, "Я": 28}

reverse_classes = {0: "А", 1: "Б", 2: "В", 3: "Г", 4: "Д", 5: "Е", 6: "Є", 7: "Ж", 8: "З", 9: "И", 10: "І", 11: "К",
                   12: "Л", 13: "М", 14: "Н", 15: "О", 16: "П", 17: "Р", 18: "С", 19: "Т", 20: "У", 21: "Ф", 22: "Х",
                   23: "Ц", 24: "Ч", 25: "Ш", 26: "Ь", 27: "Ю", 28: "Я"}

import os
import shutil
import random

# Пути к папкам train и validation
train_path = r'D:\SignLanguage\data\dataset\train'
validation_path = r'D:\SignLanguage\data\dataset\validation'

# Создание папок в папке validation
# for class_folder in os.listdir(train_path):
#     class_folder_path_train = os.path.join(train_path, class_folder)
#     class_folder_path_validation = os.path.join(validation_path, class_folder)
#
#     # Создание папок validation, если они еще не существуют
#     if not os.path.exists(class_folder_path_validation):
#         os.makedirs(class_folder_path_validation)
#
#     # Получение списка изображений в папке train
#     images = os.listdir(class_folder_path_train)
#
#     # Выбор 3 случайных изображений
#     random_images = random.sample(images, 3)
#
#     # Копирование выбранных изображений в папку validation
#     for image in random_images:
#         source_path = os.path.join(class_folder_path_train, image)
#         destination_path = os.path.join(class_folder_path_validation, image)
#         shutil.copyfile(source_path, destination_path)
#
#         # Удаление скопированных изображений из папки train
#         os.remove(source_path)
#
# print("Готово.")

# create folder for each letter in data/dataset/
for letter in classes:
    # delete the folder if it exists
    if os.path.exists(f"data/dataset/{letter}"):
        os.rmdir(f"data/dataset/{letter}")
    os.makedirs(f"data/dataset/{classes[letter]}", exist_ok=True)

video_path = "C:/Users/Admin/Downloads/Telegram Desktop/IMG_0367.MOV"
cap = cv2.VideoCapture(video_path)

h, w = 640, 480
cap.set(3, w)
cap.set(4, h)

count = 0
# read the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    # display the frame
    resized = cv2.resize(frame, (h, w))
    cv2.imshow("Frame", resized)

    if count % 10 == 0:
        letter = ""
        if letter:
            cv2.imwrite(f"data/dataset/{letter}/{len(os.listdir(f'data/dataset/{letter}'))}.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
