import numpy as np
import cv2

def reSize(image,width= None,height = None):
    if height is None:
        r = width / image.shape[1]
        dim = (width, int(image.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if width is None:
        r = height / image.shape[0]
        dim = (int(image.shape[1] * r), height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def rotate(image, angle, scale):
    (h,w) = image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(image,M,(w,h))
    return rotated

def translate(image, x, y):
    M = np.float32([1,0,x],[0,1,y])
    trans = cv2.warpAffine(image, M, (image.shape[1], image.shape[2]))
    return trans

def rectMasking(image,  mX, mY, center = None, sX = None, sY = None):
    mask = np.zeros(image.shape[:2], dtype="uint8")

    if center == "center":
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.rectangle(mask, (cX - mX, cY - mY), (cX + mX, cY + mY), 255, -1)
    else:
        cv2.rectangle(mask, (sX - mX, sY - mY), (sX + mX, sY + mY), 255, -1)

    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked

def circMasking(image,center, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius,255,-1)
    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked

def cannyDetection(image, gaus, upperBound, lowerBound):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (gaus, gaus),0)
    canny = cv2.Canny(image,lowerBound, upperBound)
    return canny

