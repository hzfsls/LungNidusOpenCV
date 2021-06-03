# 本文件中为SVM训练模块，可以训练两种纹理的提取

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from utils import loadReticularMask, loadHoneycombingMask, loadProcessedImg, savePredictImg
from tuning import trainPreprocess, areaInterval, areaSize, svm_C, svm_Gamma, coDistances, calculate

areaNum = (512 - areaSize) // areaInterval + 1


def imageMark(img):
    res = img.copy()
    areas = [
        [res[i * areaInterval:i * areaInterval + areaSize,
         j * areaInterval:j * areaInterval + areaSize]
         for j in range(areaNum)]
        for i in range(areaNum)]

    areaRatio = [[cv2.countNonZero(areas[i][j]) / (areaSize * areaSize) for j in range(areaNum)]
                 for i in range(areaNum)]

    areaRatio = np.reshape(areaRatio, (areaNum * areaNum))
    ratioThresh = 0.25
    mark = np.where(areaRatio > ratioThresh, 1, 0).astype('int32')

    return mark


def imageCoprops(img):
    res = img.copy()

    greyLevelImg = trainPreprocess(res)

    areas = [
        [greyLevelImg[i * areaInterval:i * areaInterval + areaSize,
         j * areaInterval:j * areaInterval + areaSize]
         for j in range(areaNum)]
        for i in range(areaNum)]

    nonZero = [[cv2.countNonZero(areas[i][j]) for j in range(areaNum)]
               for i in range(areaNum)]

    nonZero = np.reshape(nonZero, (areaNum * areaNum))

    zeroMask = np.where(nonZero > 0, 1, 0).astype('int32')

    # dirs = [0]
    dirs = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    # 求小区域的灰度共生矩阵
    glcms = [
        [greycomatrix(areas[i][j], coDistances, dirs, 9, symmetric=True,
                      normed=True)
         for j in range(areaNum)]
        for i in range(areaNum)]

    print("灰度共生矩阵求取完毕")

    # 求小区域的灰度矩阵属性
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    # coprops[i][j][k][d]: 第(i,j)个小区域的灰度共生矩阵中，第d个方向的第k种属性参数
    coProps = [
        [[[greycoprops(glcms[i][j], prop)[0][k]
           for k in range(len(dirs))]
          for prop in props]
         for j in range(areaNum)]
        for i in range(areaNum)]
    print("根据共生矩阵求取属性完毕")

    coProps = np.reshape(coProps, (areaNum * areaNum, len(dirs) * len(props))).astype('float32')

    return coProps, zeroMask


def calcTrainingParams(calc, imgType=1):
    if calc == 1:
        coPropsAllI = []
        trainDataXI = []
        trainDataYI = []
        for i in range(1, 21):
            print("训练图像: " + str(i))
            img = loadProcessedImg(i, imgType)
            if imgType == 0:
                hMask = loadHoneycombingMask(i, imgType)
            else:
                hMask = loadReticularMask(i, imgType)
            mark = imageMark(hMask)
            # markSave = np.reshape(mark, (areaNum, areaNum)).astype('uint8') * 255
            # markSave = cv2.resize(markSave, (areaNum * areaInterval, areaNum * areaInterval))
            # savePredictImg(markSave, i + 50, imgType)
            coPropsI, zeroMask = imageCoprops(img)
            coPropsAllI.append(coPropsI)

            # if i > 10:
            #     continue
            filteredMark = []
            filteredCoProps = []
            for k in range(areaNum * areaNum):
                if zeroMask[k] != 0:
                    filteredMark.append(mark[k])
                    filteredCoProps.append(coPropsI[k])

            filteredMark = np.array(filteredMark)
            filteredCoProps = np.array(filteredCoProps)
            trainDataXI.extend(filteredCoProps)
            trainDataYI.extend(filteredMark)

        np.save("data/coPropsAll" + str(imgType) + ".npy", np.array(coPropsAllI))
        np.save("data/trainDataX" + str(imgType) + ".npy", np.array(trainDataXI))
        np.save("data/trainDataY" + str(imgType) + ".npy", np.array(trainDataYI))

    trainDataY = np.load("data/trainDataY" + str(imgType) + ".npy")
    trainDataX = np.load("data/trainDataX" + str(imgType) + ".npy")
    coPropsAll = np.load("data/coPropsAll" + str(imgType) + ".npy")
    return coPropsAll, trainDataX, trainDataY


def SVM_train(imgType):
    # 获得参数 0为读取事先参数 1位计算获得
    coPropsAll, trainDataX, trainDataY = calcTrainingParams(calculate, imgType)

    # 训练
    mySVM = cv2.ml.SVM_create()
    mySVM.setType(cv2.ml.SVM_C_SVC)
    mySVM.setC(svm_C)
    mySVM.setKernel(cv2.ml.SVM_RBF)
    mySVM.setGamma(svm_Gamma)
    # mySVM.setKernel(cv2.ml.SVM_POLY)
    # mySVM.setDegree(0.5)
    print("开始训练")
    result = mySVM.train(trainDataX, cv2.ml.ROW_SAMPLE, trainDataY)
    if (imgType == 0):
        mySVM.save("mySVM/h_svm.mat")
    else:
        mySVM.save("mySVM/r_svm.mat")


if __name__ == "__main__":
    SVM_train(0)
    SVM_train(1)
