# 本文件中为可调参数和函数，在此进行调整来达到更好的效果

import cv2
import numpy as np

# 本文件中为可调参数和函数
# 每篇小区域的边长
areaSize = 16
# 每篇小区域之间的间隔
areaInterval = 16


#共生矩阵求取时的距离
coDistances = [4]

# 是否重新计算灰度共生矩阵参数，每次调整areaSize,areaInterval或coDistance之后需要开启
# 1为开启 0为关闭
calculate = 1

# SVM向量机的C参数
svm_C = 3
# SVM向量机的Gamma参数
svm_Gamma = 1


# 训练之前对图片进行的预处理
def trainPreprocess(img):
    res = cv2.equalizeHist(img)
    greyLevelImg = np.zeros(img.shape, np.int32)
    # 灰度等级压缩
    for i in range(8):
        low = i * 32
        high = (i + 1) * 32
        iLevelImg = cv2.inRange(res, low, high - 1)
        iLevelImg = np.where(iLevelImg, 1, 0)
        greyLevelImg += (iLevelImg * i)

    return greyLevelImg

