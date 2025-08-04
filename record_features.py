import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

features = []
labels = []

cap = cv2.VideoCapture(0)

label = input(": ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)
            features.append(data)
            labels.append(label)

    cv2.imshow("Recording...", image)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("✅ ")
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(features)
df["label"] = labels
df.to_csv("features.csv", mode='a', index=False, header=not pd.read_csv("features.csv").shape[0] if pd.io.common.file_exists("features.csv") else True)
ii

