# 本文件中为SVM预测模块，可以对图像进行预测并输出结果

import cv2
import numpy as np
from utils import loadReticularMask, loadHoneycombingMask, loadProcessedImg, savePredictImg, loadTestImg
from tuning import areaInterval, areaSize
from svm import SVM_train

areaNum = (512 - areaSize) // areaInterval + 1


def outputPredictImage(imgType, h_masks, r_masks):
    for index in range(1, 21):
        colorImg = loadTestImg(index, imgType)
        h_mask = h_masks[index - 1]
        r_mask = r_masks[index - 1]
        colorImgB = colorImg[:, :, 0]
        colorImgG = colorImg[:, :, 1]
        colorImgR = colorImg[:, :, 2]
        colorImgB = np.where(h_mask > 0, colorImgB / 2, colorImgB)
        colorImgG = np.where(r_mask > 0, colorImgG / 2, colorImgG)
        colorImgR = np.where(h_mask > 0, colorImgR / 2, colorImgR)
        colorImg[:, :, 0] = colorImgB
        colorImg[:, :, 1] = colorImgG
        colorImg[:, :, 2] = colorImgR
        savePredictImg(colorImg, index, imgType)


def SVM_predict(imgType, svmType=-1):
    if svmType == -1:
        svmType = imgType

    if svmType == 0:
        mySVM = cv2.ml.SVM_load("mysvm/h_svm.mat")
    else:
        mySVM = cv2.ml.SVM_load("mysvm/r_svm.mat")
    r_masks = []
    ious = []
    for index in range(1, 21):
        # print("预测图像: " + str(index))
        img = loadProcessedImg(index, imgType)
        # cv2.imshow('img', img)
        coPropsAll = np.load("data/coPropsAll" + str(imgType) + ".npy")
        coProps = coPropsAll[index - 1]
        _, y_pred = mySVM.predict(coProps)
        r_y = np.reshape(y_pred, (areaNum, areaNum)) * 255
        mask = np.where(img > 10, 255, 0).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        r_mask = np.zeros(img.shape, np.uint8)
        for m in range(areaNum):
            for n in range(areaNum):
                # d = (areaSize - areaInterval) // 2
                d = 0
                r_mask[m * areaInterval + d:m * areaInterval + areaSize - d,
                n * areaInterval + d:n * areaInterval + areaSize - d] = r_y[m][n]

        r_mask = cv2.copyTo(r_mask, mask)
        r_masks.append(r_mask)

        if svmType == 0:
            true_r_mask = loadHoneycombingMask(index, imgType)
        else:
            true_r_mask = loadReticularMask(index, imgType)
        i_area = np.where(r_mask > 0, true_r_mask, 0)
        u_area = np.where(r_mask == 0, true_r_mask, r_mask)

        if (cv2.countNonZero(u_area) != 0):
            ious.append(cv2.countNonZero(i_area) / cv2.countNonZero(u_area))
        else:
            ious.append(-1)
    return r_masks, ious


if __name__ == "__main__":
    # SVM_train(0)
    # SVM_train(1)

    h_masks, h_ious = SVM_predict(0, 0)

    print(h_ious)
    print("HoneyCombing IOU : " + str(np.average(h_ious)))

    r_masks, r_ious = SVM_predict(1, 1)

    print(r_ious)
    print("Reticular IOU : " + str(np.average(r_ious)))

    zero_masks = np.zeros(np.array(h_masks).shape)

    # rh_masks, _ = SVM_predict(0, 1)
    # hr_masks, _ = SVM_predict(1, 0)
    outputPredictImage(0, h_masks, zero_masks)
    outputPredictImage(1, zero_masks, r_masks)
