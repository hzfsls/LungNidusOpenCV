import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from utils import loadTestImg, loadTestSnapshot, saveSnapColor
from preprocess import lungMask


def resizeSnapshot(img, imgType):
    if imgType == 0:
        rWidth, rHeight = 534, 394
    else:
        rWidth, rHeight = 470, 348
    resizeImg = cv2.resize(img, (rWidth, rHeight))
    if (rWidth > 512):
        startXPos, endXPos = (rWidth - 512) // 2, (512 + rWidth) // 2
        resizeImg = resizeImg[:, startXPos:endXPos]
        rWidth = 512

    startXPos, endXPos = (512 - rWidth) // 2, (512 + rWidth) // 2
    startYPos, endYPos = (512 - rHeight) // 2, (512 + rHeight) // 2

    outputImg = np.zeros((512, 512, 3), np.uint8)
    outputImg[startYPos:endYPos, startXPos:endXPos] = resizeImg
    # cv2.imshow('outputImg', outputImg)
    return outputImg


def getHueImage(img):
    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hueImg = hlsImg[:, :, 0]
    return hueImg


def hueImageProcess(hueImg, color):
    low = 0
    high = 255

    if color == 'yellow':
        low = 20
        high = 30
    elif color == 'green':
        low = 30
        high = 60
    elif color == 'purple':
        low = 120
        high = 150
    _, lowThresh = cv2.threshold(hueImg, low, 255, cv2.THRESH_BINARY_INV)
    _, highThresh = cv2.threshold(hueImg, high, 255, cv2.THRESH_BINARY_INV)
    thresh = lowThresh - highThresh
    result = cv2.copyTo(hueImg, thresh)
    return result


if __name__ == "__main__":
    for imgType in range(2):
        for i in range(1, 21):
            img = loadTestImg(i, imgType)
            lung = lungMask(img)

            snapImg = loadTestSnapshot(i, imgType)
            resizedSnap = resizeSnapshot(snapImg, imgType)

            maskedSnap = cv2.copyTo(resizedSnap, lung)

            hueImg = getHueImage(maskedSnap)

            yellowImg = hueImageProcess(hueImg, 'yellow')
            yellowMask = np.where(yellowImg > 0, 255, 0)
            saveSnapColor(yellowMask, i, imgType, 'yellow')
            greenImg = hueImageProcess(hueImg, 'green')
            greenMask = np.where(greenImg > 0, 255, 0)
            purpleImg = hueImageProcess(hueImg, 'purple')
            purpleMask = np.where(purpleImg > 0, 255, 0)
            cv2.imshow('yellowMask', yellowMask.astype('uint8'))
            cv2.imshow('greenMask', greenMask.astype('uint8'))
            cv2.imshow('purpleMask', purpleMask.astype('uint8'))
            cv2.waitKey(0)

            saveSnapColor(purpleMask, i, imgType, 'purple')

