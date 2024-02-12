# Ukrainian Sign Alphabet Recognition

This project aims to recognize Ukrainian Sign Alphabet using Convolutional Neural Networks (CNNs). The provided model architecture and dataset allow you to train a model capable of identifying Ukrainian sign language gestures from video input.

## Model Architecture

The model architecture is based on a sequential CNN design, suitable for processing temporal data such as video sequences. Here's an overview of the model structure:

1. **Convolutional Layers**: These layers extract features from the input video frames.
   - Convolutional layers with ReLU activation function.
   - MaxPooling layers for down-sampling the spatial dimensions.

2. **Dropout Layer**: Regularization technique to prevent overfitting.

3. **Dense Layers**: Fully connected layers for classification.
   - ReLU activation functions for intermediate layers.
   - Softmax activation function for the output layer to generate class probabilities.

## Dataset

The dataset provided contains video samples of Ukrainian Sign Alphabet gestures. You can find the dataset in the following Google Drive link: [Ukrainian Sign Alphabet Dataset](https://drive.google.com/file/d/11qAmGBbme2bLd3XtuwepaNvJP9e77dde/view?usp=sharing).

## Usage

1. **Model Training**:
   - Modify the paths in `/local/user/SignLanguage/data/paths.py` to point to your dataset.
   - Use the provided model architecture to train your model on the dataset.

2. **Demo and Application**:
   - Run `main.py` to build a demo using your trained model for recognizing sign language gestures from video input.
   - Run `app.py` to start a streaming service on your localhost, allowing real-time recognition of sign language gestures.

## GIF Demo

You can view a demonstration of the project usage in the following GIF:

![Ukrainian Sign Alphabet Recognition Demo](demo.gif)

**Note:** This demo GIF was captured on a laptop with Intel Core i7 and 16GB RAM. Due to these specifications, the streaming capability of the application could achieve around 5 FPS.

## Note

Make sure to customize the paths and configurations according to your setup before running the code.

Thank you for using the Ukrainian Sign Alphabet Recognition project! If you have any questions or feedback, feel free to contact us.
