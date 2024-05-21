import cv2
import numpy as np


def getContours(img, CThr=[50, 50], showCanny=False, minArea=1000, filter=0, draw=False):
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image to be a gray color
    BlurImg = cv2.GaussianBlur(GrayImg, (3, 3), 1)  # smooth the image by blurring to remove any noise, kernel size(1,1)
    CannyImg = cv2.Canny(BlurImg, CThr[0], CThr[1])  # detect edges using canny
    kernel = np.ones((3, 3))
                           # perform a dilation + erosion to close gaps in between object edges
    imgDial = cv2.dilate(CannyImg, kernel, iterations=3)  # to increase the object's size
    imgThres = cv2.erode(imgDial, kernel, iterations=2) # to decrease the object's size
    imgThres_res = cv2.resize(imgThres, (700,490))
    if showCanny: cv2.imshow('Canny', imgThres_res)  # show canny
    contours, hiearchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)  # find the area of the contours
        if area > minArea:  # filter based on the minimum
            per = cv2.arcLength(i, True)  # find perimeter , true - closed
            approx = cv2.approxPolyDP(i, 0.02 * per, True)  # approximate the contours shape and get corner points
            bbox = cv2.boundingRect(approx)  # rectangles of the approx points
            if filter > 0:  # so we can work only on rectangles

                if len(approx) == filter:  # do the filter
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:  # no filter
                finalContours.append([len(approx), area, approx, bbox, i])
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)  # sort based on area in descending order
    if draw:  # draw the contours
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0.0, 255), 3)
    return img, finalContours


# to reorder the coordinate point of the boarders(tl,tr,bl,br)
def reorder(thePoints):  # take the points i want to reorder them in shape(4,1,2)
    #print(thePoints.shape)  # the shape is (4,1,2)
    # declare array of shape like the input points where I store the reordered points
    # then return them in the same shape of the inputs which is (4,1,2)
    thePointsNew = np.zeros_like(thePoints)
    thePoints = thePoints.reshape((4, 2))  # reshaping to (4,2) as 4 is the number of point, 2 is each point has x,y
    add = thePoints.sum(1)  # sum along axis 1 to get points at (0,0) & (w,h)
    thePointsNew[0] = thePoints[
        np.argmin(add)]  # point at the minimum index is stored at index 0 of the new points (0,0)
    thePointsNew[3] = thePoints[
        np.argmax(add)]  # point at the maximum index is stored at index 3 of the new points (w,h)
    diff = np.diff(thePoints, axis=1)  # difference along axis 1 to get point at (w,0) & (0,h)
    thePointsNew[1] = thePoints[
        np.argmin(diff)]  # point at the minimum index is stored at index 1 of the new points (w,0)
    thePointsNew[2] = thePoints[
        np.argmax(diff)]  # point at the maximum index is stored at index 2 of the new points (0,h)
    return thePointsNew


def warpImage(img, points, w, h, pad=50):
    # print(points)

    points = reorder(points)  # reorder points

    pts1 = np.float32(points)  # the reordered points
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # pattern of points
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # matrix of the points in pattern
    Warpimg = cv2.warpPerspective(img, matrix, (w, h))  # warp the image, (w,h) is the size of the output image
    # padding so we can define the paper only without any corner pixels not of the paper
    Warpimg = Warpimg[pad:Warpimg.shape[0] - pad, pad:Warpimg.shape[1] - pad]
    return Warpimg


# to get the distance (length) of the line in the object using the formula a^2+b^2 = c^2
# pts1 - x1, y1 , pts2 - x2,y2
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
