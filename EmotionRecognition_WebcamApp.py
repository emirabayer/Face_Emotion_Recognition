import cv2
import numpy as np
import time

final_model = 'face_emotion_recognition.h5'

path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (255,255,255)
img = np.zeros((500,500))
text = "Text"
(text_width, text_height) = cv2.getTextSize(text,font,fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)





def list_webcams(max_index=10, backend_timeout=2):
    print("Testing webcam indices...")
    available_cams = []
    
    # Try multiple backends for robustness
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for index in range(max_index):
        for backend in backends:
            backend_name = {cv2.CAP_DSHOW: "CAP_DSHOW", cv2.CAP_MSMF: "CAP_MSMF", cv2.CAP_ANY: "CAP_ANY"}[backend]
            print(f"Testing index {index} with backend {backend_name}...")
            
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
                
            # Give webcam time to initialize
            start_time = time.time()
            while time.time() - start_time < backend_timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Webcam found at index {index} (backend {backend_name}): {width}x{height}")
                    available_cams.append((index, backend_name))
                    break
                time.sleep(0.1)
                
            cap.release()
            cv2.destroyAllWindows()
            
            # Small delay to ensure release
            time.sleep(0.1)
    
    return available_cams


cams = list_webcams()
if cams:
    print(f"\nAvailable webcam indices: {cams}")
else:
    print("\nNo webcams found.")


cam = int(input("Select webcam index: "))
cap = cv2.VideoCapture(cam)





while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in faces:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew] # cropping the face
    
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0) # adding fourth dimension
    final_image - final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    Predictions = final_model.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    if (np.argmax(Predictions)=="Angry"):
        status = "Angry"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Disgust"):
        status = "Disgust"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Fear"):
        status = "Fear"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Happy"):
        status = "Happy"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Sad"):
        status = "Sad"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Disgust"):
        status = "Disgust"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif (np.argmax(Predictions)=="Surprise"):
        status = "Surprise"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    else:
        status = "Neutral"

        x1,y1,w1,h1 = 0,0,175,75
        # black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2))
        cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))



    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
