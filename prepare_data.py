import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

rectangles = []
abc = "abcdefghijklmnopqrstuvwxyz"


def getContours(img, output, path, img_to_slice):
    rectangles.clear()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        tk3 = cv2.getTrackbarPos("t3", "Params")
        if area > tk3:
            # cv2.drawContours(output, cnt, -1, (0, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x_, y_, w_, h_ = cv2.boundingRect(approx)
            rectangles.append((x_, y_, w_, h_))
            cv2.rectangle(output, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

            filename = f"{count}.jpg"
            letter = img_to_slice[y_:y_+h_, x_:x_+w_]
            resized_letter = cv2.resize(letter, (64,64))
            print(f"{path}/{filename}")
            cv2.imwrite(f"{path}/{filename}", resized_letter)
            # cv2.putText(output, str(count), (x_,y_), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0,255), 1)
            count += 1

    return output


for char in abc:
    path = f"Training/{char}"
    os.mkdir(path)
    strip_img = cv2.imread(f"Training\\{char}_strip.jpg")

    gray = cv2.cvtColor(strip_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
    imgCanny = cv2.Canny(thresh, threshold1=234, threshold2=137)

    kernel = np.ones((2, 2))
    imDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imCopy = strip_img.copy()
    cont = getContours(imDil, imCopy, path, strip_img)
    cv2.imshow("lol", imCopy)
    cv2.waitKey(0)





