import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture("00-basketball-shot-predictor/Videos/1.mp4")

# Create the color finder object
my_color_finder = ColorFinder(False)
hsv_vals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Variables
pos_list_x = []
pos_list_y = []
x_list = [item for item in range(0, 1300)]
prediction = False

while True:

    # Grab the image
    success, img = cap.read()
    #img = cv2.imread("00-basketball-shot-predictor/Ball.png")
    img = img[0:900, :]

    # Find the Color Ball
    img_color, mask = my_color_finder.update(img, hsv_vals)

    # Fİnd location of the ball
    img_contours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        pos_list_x.append(contours[0]["center"][0])
        pos_list_y.append(contours[0]["center"][1])

    if len(pos_list_x) > 7:

        # Polynomial Regression y = Ax^2 + Bx + C
        # Find the Coefficients
        A, B, C = np.polyfit(pos_list_x, pos_list_y, 2)

        for i, (pos_x, pos_y) in enumerate(zip(pos_list_x, pos_list_y)):
            pos = (pos_x, pos_y)
            cv2.circle(img_contours, (pos), radius=10, color=(0, 255, 0), thickness=-1)

            if i == 0:
                cv2.line(img_contours, pos, pos, color=(0, 255, 0), thickness=5)
            else:
                cv2.line(img_contours, pos, (pos_list_x[i-1], pos_list_y[i-1]), color=(0, 255, 0), thickness=5)

        for x in x_list:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(img_contours, (x, y), radius=2, color=(255, 0, 255), thickness=-1)

        if len(pos_list_x) < 10:

            # Prediction
            # X values are 330 to 430 - Y value is constant and it is 590
            a = A
            b = B
            c = C - 590

            x = int((-b - np.math.sqrt(b ** 2 - (4*a*c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(img_contours, "Basket!", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(img_contours, "No Basket.", (50, 150), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Dİsplay
    img_contours = cv2.resize(img_contours, (0, 0), None, 0.7, 0.7)
    
    #cv2.imshow("Image", img)
    cv2.imshow("Image color", img_contours)
    
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()