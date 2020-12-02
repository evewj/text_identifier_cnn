#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import imutils

def deskew(img, pts1, pts2, shape):
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    M, mask = cv2.findHomography(pts1,pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,shape)

    return dst


def find_order_pts(points):
    points = points[:,0,:]
    #points array(array(x,y))
    e1 = np.mean(points[:,0])
    e2 = np.mean(points[:,1])

    low_low = None
    low_high = None
    high_low = None
    high_high = None

    for i in range(4):
        p = points[i]
        if p[0] > e1:
            if p[1] > e2:
                high_high = p
            else:
                high_low = p
        else:
            if p[1] > e2:
                low_high = p
            else:
                low_low = p

    if low_low is None or low_high is None or high_low is None or high_high is None:
        return points
    else:
        return np.array([low_low,low_high,high_high,high_low])

def parse_image(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv', hsv_frame)
    shape = (1280, 720)
    shape_3d = (1280, 720, 3)

    normal_lower_hsv = np.array([0,0,86])
    normal_upper_hsv = np.array([141,35,125])

    darker_lower_hsv = np.array([0,0,148])
    darker_upper_hsv = np.array([186,18,202])

    img_mask_1 = cv2.inRange(hsv_frame, normal_lower_hsv, normal_upper_hsv)
    img_mask_1 = cv2.resize(img_mask_1, shape )

    img_mask_2 = cv2.inRange(hsv_frame, darker_lower_hsv, darker_upper_hsv)
    img_mask_2 = cv2.resize(img_mask_2, shape )

    img_mask = img_mask_1 + img_mask_2

    img = cv2.resize(img, shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(img_mask, 25, 300, 300)
    cv2.imshow('mask', gray)
    edged = cv2.Canny(gray, 30, 200)
    #cv2.imshow('edges', edged )
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt1 = None

    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        cv2.drawContours(img, [approx], -1, (255, 255, 255), 3)
        #print(len(approx))
        if len(approx) == 4:
            if screenCnt1 is None:
                screenCnt1 = approx
                break

    if screenCnt1 is None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    else:
        screenCnt1 = find_order_pts(screenCnt1)
        #cv2.imshow('edges', edged )
        #cv2.imshow('mask', gray)

        detected = deskew(img, screenCnt1, [[0,0],[0,1800],[600,1800],[600,0]], (600,1800))

        location = detected[:1800-298-260,:]
        plate = detected[1800-298-260:1800-260,:]

        plate_letters = [plate[50:260,30:170],plate[50:260,130:270], plate[50:260,330:470], plate[50:260,430:570]]
        location_letters = [location[650:1100,0:300],location[650:1100,300:]]

        letters = []
        model_shape = ( 100, 150)

        try:
            for j in range(2):
                letter = location_letters[j]
                letter = cv2.resize(letter, model_shape)
                cv2.imshow('location'+str(j), letter)
                letters.append(letter)

            for j in range(4):
                letter = plate_letters[j]
                letter = cv2.resize(letter, model_shape)
                cv2.imshow('plate' + str(j), letter)
                letters.append(letter)
        except Exception as e:
            print(e)
            return None

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return letters


PATH = os.path.dirname(os.path.abspath(__file__))+'/Training data/*'
OUT_PATH = os.path.dirname(os.path.abspath(__file__))+ '/Training output/Image_'
files = glob.glob(PATH)
print(files)
count = 0
for file in files:
    letters = parse_image(file)
    print(file)
    if letters is not None:
        for i in range(0,6):
            let = letters[i]
            #print(let.shape)
            name = os.path.join(OUT_PATH + str(count) +'.png')
            print(name)
            if not cv2.imwrite(name,let):
                print(file + ' failed')
            count = count+1
