import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    p_time = time.time()

    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for lm_id, lm in enumerate(hand_lms.landmark):
                # print(lm_id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(lm_id, cx, cy)

                if lm_id == 0:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 / (c_time - p_time)

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
