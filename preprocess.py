# 本文件中为预处理模块，可以将肺实质进行提取

import cv2
import numpy as np
import matplotlib.pyplot as plt


def tubeMask(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape

    _, binaryImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    newBinaryImg = binaryImg.copy()
    newMask = cv2.bitwise_not(newBinaryImg, newBinaryImg)
    retval, labels = cv2.connectedComponents(newMask)
    unique, counts = np.unique(labels, return_counts=True)
    connectedMask = np.zeros(binaryImg.shape, np.uint8)
    for i in range(0, retval):
        if 600 < counts[i] < 700:
            tmpImg = np.where(labels == i, 255, 0)
            connectedMask += tmpImg.astype('uint8')
    connectedMask = cv2.morphologyEx(connectedMask, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    return connectedMask


def lungMask(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape
    _, binaryImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    openedImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    borderImg = cv2.copyMakeBorder(openedImg, 200, 200, 200, 200, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    borderImg = cv2.morphologyEx(borderImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200)))

    mask = borderImg[201:201 + width, 201:201 + height]
    return mask


def processImage(index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./test/" + typeStr + indexStr + "00001.jpg"
    storepath = "./preprocess/" + typeStr + indexStr + "00001.jpg"
    storepathCmp = "./preprocess/" + typeStr + "compare/" + indexStr + "00001.jpg"

    print("Processing " + filepath)

    img = cv2.imread(filepath)
    tubemask = tubeMask(img)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, _ = img.shape

    _, binaryImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    openedImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    borderImg = cv2.copyMakeBorder(openedImg, 200, 200, 200, 200, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    borderImg = cv2.morphologyEx(borderImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200)))

    mask = borderImg[201:201 + width, 201:201 + height]

    maskBinaryImg = cv2.copyTo(binaryImg, mask)

    maskGrayImg = cv2.copyTo(grayImg, mask)

    _, newBinaryImg = cv2.threshold(maskGrayImg, 200, 255, cv2.THRESH_BINARY)

    newBinaryImg += tubemask

    processedImg = cv2.morphologyEx(newBinaryImg, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

    processedImg = cv2.morphologyEx(processedImg, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    newMask = cv2.bitwise_not(processedImg, processedImg)
    finalImg = cv2.copyTo(maskGrayImg, newMask)

    _, binaryImg = cv2.threshold(finalImg, 1, 255, cv2.THRESH_BINARY)

    retval, labels = cv2.connectedComponents(binaryImg)
    unique, counts = np.unique(labels, return_counts=True)
    connectedMask = np.zeros(binaryImg.shape, np.uint8)
    for i in range(0, retval):
        if 100000 > counts[i] > 2000:
            tmpImg = np.where(labels == i, 255, 0)
            connectedMask += tmpImg.astype('uint8')

    connectedMask = cv2.morphologyEx(connectedMask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

    finalImg = cv2.copyTo(grayImg, connectedMask)
    imgCmp = np.hstack((grayImg, finalImg))

    cv2.imwrite(storepath, finalImg)
    cv2.imwrite(storepathCmp, imgCmp)


if __name__ == "__main__":
    processImage(10, 0)
    cv2.waitKey(0)
    # for i in range(1, 21):
    #     processImage(i, imgType=0)
    # for i in range(1, 21):
    #     processImage(i, imgType=1)
