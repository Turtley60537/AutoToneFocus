# -*- coding: utf-8 -*-


import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("./images/focus_flower.jpg", 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (256, 256))

    cv2.imwrite("gray1.jpg", gray)

    # plt.imshow(gray, cmap = 'gray')

    height, width = gray.shape

    d    = 32
    lenH = int(height/d)
    lenW = int(width/d)

    for i in range(lenH):
        for j in range(lenW):
            i0 = i*d
            j0 = j*d
            segImg = gray[i0:i0+d, j0:j0+d]
            # cv2.imwrite("./test-%s-%s.jpg" % (i,j), segImg)
            
            # 高速フーリエ変換(2次元)
            f = np.fft.fft2(segImg)
            # 零周波数成分を配列の左上から中心に移動
            fshift =  np.fft.fftshift(f)
            mgnSpct = 20*np.log(np.abs(fshift))
            # mgnSpct = np.log(np.abs(fshift))

            plt.subplot(lenH, lenW, j+i*lenW+1)
            plt.imshow(mgnSpct, cmap = 'gray')
            plt.title("%s %s" % (j, i))
            plt.xticks([])
            plt.yticks([])


    # # 高速フーリエ変換(2次元)
    # f = np.fft.fft2(gray)
    # # 零周波数成分を配列の左上から中心に移動
    # fshift =  np.fft.fftshift(f)
    # mgnSpct = 20*np.log(np.abs(fshift))



    # グレースケールの表示
    # plt.subplot(233)
    # plt.imshow(gray, cmap = 'gray')
    # plt.title('Input Image')
    # plt.xticks([])
    # plt.yticks([])

    # 全体画像についてのフーリエ
    # plt.subplot(234)
    # plt.imshow(mgnSpct, cmap = 'gray')
    # plt.title('Magnitude Spectrum')
    # plt.xticks([])
    # plt.yticks([])

    plt.show()

    # # 零周波数成分を中心に移動した結果を表示
    # print(fimg)
    # cv2.imshow(fimg)

if __name__ == "__main__":
    main()