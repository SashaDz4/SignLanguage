import cv2

from model import LetterDetector
from utils import process_data, preparation_data


# Extract Feature for Training
training_paths = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/training/"
training_csv_path = "data/hands_SIBI_training.csv"
process_data(training_paths, training_csv_path)

# Extract Feature for Validation
validation_paths = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/validation/"
validation_csv_path = "data/hands_SIBI_validation.csv"
process_data(validation_paths, validation_csv_path)

x_train, y_train, x_test, y_test = preparation_data()

# Directly from Imageset Dataset Testing
# Load Image and do Feature Extraction
path_to_image = "/home/oleksandr/Downloads/test_img_E.jpg"
path_to_image = "/home/oleksandr/Downloads/archive (3)/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/SIBI_datasets_LEMLITBANG_SIBI_R_90.10_V02/test/Q (4).jpg"

model = LetterDetector(x_train.shape[1:3])
# model.train(x_train, y_train, x_test, y_test, epochs=100, batch_size=32)
# model.summary()
# model.save("model1.h5")
model.load("models/model1.h5")

input_IMG = cv2.imread(path_to_image)

# Print the Prediction
predict = model.predict_classes(input_IMG)
print("Prediction : ", predict)

input_IMG = cv2.resize(input_IMG, (640, 480))
cv2.putText(input_IMG, predict, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Prediction", input_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

