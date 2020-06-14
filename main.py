import numpy as np
import cv2
import imageio
import skimage
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from utils import preprocess_input, get_labels
emotion_classifier=load_model('./model/emotion_classification', compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_labels = get_labels()
detector = MTCNN()
cap = cv2.VideoCapture(0)
emotion_window=[]
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            x1 = bounding_box[0]
            y1 = bounding_box[1]
            x2 = bounding_box[0] + bounding_box[2]
            y2 = bounding_box[1] + bounding_box[3]
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0,155,255),
                          2)
            
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            
            gray_face = gray_face [y1:y2, x1:x2]
            try:
              gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
              continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            print(emotion_prediction)
            print(emotion_probability)
            print(emotion_text)
            cv2.putText(frame, emotion_text, (300,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    #display resulting fram
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
