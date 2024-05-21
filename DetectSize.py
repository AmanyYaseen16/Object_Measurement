import cv2
import PrFunctions
import numpy as np

webcam = False
path = 'CardT.jpg'
cap = cv2.VideoCapture(0)
scale = 3
wP = 212 * scale
hP = 300 * scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
        imgContours, conts = PrFunctions.getContours(img, showCanny=True, minArea=50000, filter=4)
        if len(conts) != 0:
            biggest = conts[0][2]
            # print(biggest)

            Warpimg = PrFunctions.warpImage(img, biggest, wP, hP)
            cv2.imshow('Warped Image', Warpimg)  # show warped image
            ImgContours2, conts2 = PrFunctions.getContours(Warpimg,
                                                           minArea=2000, filter=4,
                                                           CThr=[50, 50], draw=False)
            if len(conts) != 0:
                for obj in conts2:
                    cv2.polylines(ImgContours2, [obj[2]], True, (255, 0, 0), 2)

                    nPoints = PrFunctions.reorder(obj[2])
                    nW = round((PrFunctions.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
                    nH = round((PrFunctions.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
                    cv2.arrowedLine(ImgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[1][0][0], nPoints[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)  # for drawing arrows , 3:thickness , 8:type line
                    cv2.arrowedLine(ImgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                    (nPoints[2][0][0], nPoints[2][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    x, y, w, h = obj[3]
                    # put numbers text on the arrows
                    cv2.putText(ImgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
                    cv2.putText(ImgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1.5,
                                (255, 0, 255), 2)   # {} place holder
            cv2.imshow('A4', ImgContours2)

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Original', img)
        if cv2.waitKey(1) & 0xFF == ord('m'):  # keep showing window until 'm' is pressed.
            break

cap.release()
cv2.destroyAllWindows()
