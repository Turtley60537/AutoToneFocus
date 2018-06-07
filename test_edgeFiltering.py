# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

def median_filter(src, ksize):
    # 畳み込み演算をしない領域の幅
    d = int((ksize-1)/2)
    h, w = src.shape[0], src.shape[1]
    
    # 出力画像用の配列（要素は入力画像と同じ）
    dst = src.copy()

    for y in range(d, h - d):
        for x in range(d, w - d):
            # 近傍にある画素値の中央値を出力画像の画素値に設定
            dst[y][x] = np.median(src[y-d:y+d+1, x-d:x+d+1])

    return dst

def main():
    img = cv2.imread("./images/focus_image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = median_filter(gray, ksize=3)

    plt.subplot(111)
    plt.imshow(gray, cmap = 'gray')
    plt.title("gray")
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    main()