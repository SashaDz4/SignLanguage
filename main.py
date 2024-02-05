import cv2
from model import LetterDetector


path_to_image = "/home/oleksandr/Downloads/test_img_E.jpg"

model = LetterDetector((63, 1))
model.load("models/model1.h5")

input_IMG = cv2.imread(path_to_image)

# Print the Prediction
prediction = model.predict_classes(input_IMG)
print("Prediction : ", prediction)

input_IMG = cv2.resize(input_IMG, (640, 480))
cv2.putText(input_IMG, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Prediction", input_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

