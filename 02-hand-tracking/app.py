import cv2
import time
import HandTrackingModule as Htm

cap = cv2.VideoCapture(0)
detector = Htm.HandDetector()

while True:
    p_time = time.time()

    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        print(lm_list[4])

    c_time = time.time()
    fps = 1 / (c_time - p_time)

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
