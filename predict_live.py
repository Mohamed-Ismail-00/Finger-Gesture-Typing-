import cv2
import mediapipe as mp
import pickle
import numpy as np


with open("model.pkl", "rb") as f:
    model = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


typed_text = ""
current_letter = ""  

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
                data.extend([lm.x, lm.y])

            prediction = model.predict(np.array(data).reshape(1, -1))
            current_letter = prediction[0]

            # عرض الحرف المتوقع
            cv2.putText(image, f'Letter: {current_letter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            break  # نستخدم يد واحدة فقط

    # عرض النص المكتوب حتى الآن
    cv2.putText(image, f'Typed: {typed_text}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Finger Detection - Type Your Name", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # للخروج
        break
    elif key == ord(' '):  # زر المسطرة -> يسجل الحرف الحالي
        if current_letter != "":
            typed_text += current_letter
    elif key == 8:  # Backspace لمسح آخر حرف
        typed_text = typed_text[:-1]
    elif key == 13:  # Enter لإضافة مسافة بين الأسماء
        typed_text += " "

cap.release()
cv2.destroyAllWindows()
