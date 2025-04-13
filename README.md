# Face Emotion Recognition

This project is a real-time application for detecting and classifying human facial emotions in video files. It utilizes a pre-trained MobileNetV2 convolutional neural network (CNN) to recognize seven distinct emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The system combines OpenCV for video processing and face detection, with TensorFlow/Keras for deep learning-based emotion classification.

<p align="center">

_See the confusion matrix and test scores at the end of this README_
</p>


## Features
- **Facial Emotion Detection**: Identifies faces in video frames using Haar cascades.
- **Real-Time Emotion Classification**: Uses a fine-tuned MobileNetV2 model to predict emotions on detected faces.
- **Visual Overlay**: Displays emotion labels on each detected face with bounding boxes and a status indicator.
- **Robust to Video Quality**: Preprocessing pipeline ensures consistent input quality for the model.
- **FPS-Aware Playback**: Automatically adjusts playback speed to maintain frame synchronization with original video FPS.

## How It Works
1. **Face Detection**:
   - Haar cascade classifier (haarcascade_frontalface_default.xml) is used to detect faces in each frame.
   - Detected face regions are cropped and resized to 224x224 pixels for classification.

2. **Image Preprocessing**:
   - RGB normalization to a [0, 1] range.
   - Reshaping to match the model input shape: (1, 224, 224, 3).

3. **Emotion Classification**:
   - A pre-trained MobileNetV2 model (fine-tuned for 7 emotion classes) is used for prediction.
   - The class with the highest probability is selected as the predicted emotion.

4. **Output**:
   - Emotion labels are rendered above detected faces.
   - A status box shows the current detected emotion for each frame.

## Model Details
- **Base Architecture**: MobileNetV2 (pre-trained on ImageNet)
- **Adaptation**: Top layers replaced with a custom classification head for 7 emotion classes.
- **Input Size**: 224x224x3
- **Output Layer**: Softmax activation with 7 output units.
- **File**: face_emotion_recognition.h5 (trained model required to run the app)

## Usage
1. Place the .mp4 video file in the working directory.
2. Ensure the model file face_emotion_recognition.h5 is present.
3. Update the video path inside EmotionRecognition_VideoApp.py if necessary.
