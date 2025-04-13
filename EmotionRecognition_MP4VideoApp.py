import cv2
import numpy as np
import tensorflow as tf

# update the following path to your MP4 file
video_path = "expressions.mp4"

# Load the trained model
final_model = tf.keras.models.load_model('face_emotion_recognition.h5')
classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceCascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml")

# Font and UI setup
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)
img = np.zeros((500, 500))
text = "Text"
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}. Check the path and file.")

# Get video FPS for accurate playback
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS unavailable
frame_delay = int(1000 / fps)  # Delay in ms (e.g., 33ms for 30 FPS)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Reached end of video. Exiting...")
        break  # Exit instead of looping

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    status = "No Face"

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        try:
            final_image = cv2.resize(roi_color, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            predictions = final_model.predict(final_image, verbose=0)
            pred_class = np.argmax(predictions)
            status = classes[pred_class]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        # Small text above face rectangle
        cv2.putText(frame, status, (x+w+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))  # Extra rectangle (matches original)

        # Top-left status box
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('Face Emotion Recognition', frame)
    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Face Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break  # Exit on 'q' or window close

cap.release()
cv2.destroyAllWindows()