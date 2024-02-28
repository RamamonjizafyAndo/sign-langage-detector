import os
import mediapipe as mp

import cv2



DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(-1)
height = 750
width = 1200
dim = (width, height)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        
        if ret==True:
            
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.putText(frame, 'Cliquez sur le bouton "Q" et ne bougez plus', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('frame', frame)
        if cv2.waitKey(5) == ord('q'):
            break
    
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j),'{}.jpg'.format(counter)), frame)
            counter += 1

cap.release()
cv2.destroyAllWindows()
