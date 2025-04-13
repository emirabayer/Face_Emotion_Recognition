# Face Emotion Recognition

This project is a real-time application for detecting and classifying human facial emotions in video files. It utilizes a pre-trained MobileNetV2 convolutional neural network (CNN) to recognize seven distinct emotional states: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The system combines OpenCV for video processing and face detection, with TensorFlow/Keras for deep learning-based emotion classification.

<p align="center">
![emotion_recognition](https://github.com/user-attachments/assets/8410a9e5-6581-4972-8e84-a1aac027d56a)
   
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

## Demonstrated Skills
   - Proficient use of deep learning libraries (TensorFlow/Keras) for model deployment.
   - Understanding of CNN architectures and transfer learning (MobileNetV2).
   - Application of real-time image classification in a user-facing system.
   - Experience with video and face processing using OpenCV.
   - Practical implementation of end-to-end ML systems from training to deployment.

## Scores
![image](https://github.com/user-attachments/assets/e2849ff1-ba1f-4570-b891-75754d66dd19)
![image](https://github.com/user-attachments/assets/fc44eb23-5952-4502-bfbc-b97da561fcbe)
