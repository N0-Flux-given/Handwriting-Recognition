import cv2
import numpy as np
import matplotlib.pyplot as plt


def OnTrackbarChange(a):
    pass


cv2.namedWindow("Params")
cv2.resizeWindow("Params", 640, 240)
cv2.createTrackbar("t1", "Params", 234, 255, OnTrackbarChange)
cv2.createTrackbar("t2", "Params", 137, 255, OnTrackbarChange)
cv2.createTrackbar("t3", "Params", 29, 90000, OnTrackbarChange)

img = cv2.imread("sample2.jpg")
blur = cv2.GaussianBlur(img, (55, 55), 1)
cv2.imshow("Original", blur)
x = img.shape[1]
y = img.shape[0]
cv2.waitKey(0)
rectangles = []



def getContours(img, output, rectangles):
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
            # cv2.putText(output, str(count), (x_,y_), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0,255), 1)
            count += 1

    return output

lines = []
def sortRectangles():
    line = []
    delta = 5 # 5 pixel distance threshold
    minSwallowedLetters = 8 #
    appended = False
    counter = 0
    for i in range(y):    # main scan line
        distances = [(i - x[1], x) for x in rectangles]
        for distance in distances:
            if distance[0] < delta:
                line.append(distance[1])
                appended = True
            elif appended and counter >= 5:
                counter = 0
                break
        if appended:
            counter += 1

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
# detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(thresh, [c], -1, (0, 0, 0), 2)

cv2.imshow("THreshold", thresh)
cv2.waitKey(0)

kernel = np.ones((1,85),np.uint8)
dilation = cv2.erode(thresh,kernel,iterations = 1)
cv2.imshow("Line dialation", dilation)
cv2.waitKey(0)

line_rects = []
while True:
    th1 = cv2.getTrackbarPos("t1", "Params")
    th2 = cv2.getTrackbarPos("t2", "Params")

    dilCanny = cv2.Canny(dilation, threshold1=th1, threshold2=th2)
    imCopy = img.copy()
    dilatedCont = getContours(dilCanny, imCopy, line_rects)
    cv2.imshow("Line contour boxes", imCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(line_rects)

# cv2.imshow("lol", thresh)
# cv2.waitKey(0)
# cv2.waitKey(0)
# if cv2.waitKey(1) & 0xFF == ord('q'):


while True:
    th1 = cv2.getTrackbarPos("t1", "Params")
    th2 = cv2.getTrackbarPos("t2", "Params")
    imgCanny = cv2.Canny(thresh, threshold1=th1, threshold2=th2)

    kernel = np.ones((2, 2))
    imDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imCopy = img.copy()
    # cv2.imshow("lol", imDil)
    cont = getContours(imDil, imCopy, rectangles)
    # sortRectangles()
    count = 0
    for rectangle in rectangles:
        cv2.putText(cont, str(count), (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0, 255), 1)
        count += 1


    cv2.imshow("lol", cont)
    print(rectangles)
    print("length: ", len(rectangles))
    cv2.waitKey(0)


    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# https://www.youtube.com/watch?v=Fchzk1lDt7Q
