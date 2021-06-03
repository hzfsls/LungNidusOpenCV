import numpy as np
import cv2


def loadTestImg(index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./test/" + typeStr + indexStr + "00001.jpg"
    img = cv2.imread(filepath)
    return img


def loadTestSnapshot(index, imgType):
    indexStr = str(index)
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./test/" + typeStr + "snapshot" + indexStr + ".jpg"
    img = cv2.imread(filepath)
    return img


def saveSnapColor(img, index, imgType, color='yellow'):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./snapcolor/" + typeStr + indexStr + color + ".jpg"
    cv2.imwrite(filepath, img)


def savePredictImg(img, index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./final/" + typeStr + indexStr + "0001.jpg"
    cv2.imwrite(filepath, img)


def loadProcessedImg(index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    filepath = "./preprocess/" + typeStr + indexStr + "00001.jpg"
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img


def loadHoneycombingMask(index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    color = 'purple'
    filepath = "./snapcolor/" + typeStr + indexStr + color + ".jpg"
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img


def loadReticularMask(index, imgType):
    indexStr = str(index).rjust(2, '0')
    if imgType == 0:
        typeStr = "honeycombing/"
    else:
        typeStr = "reticular/"

    color = 'yellow'
    filepath = "./snapcolor/" + typeStr + indexStr + color + ".jpg"
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img


