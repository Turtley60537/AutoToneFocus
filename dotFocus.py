# -*- coding: utf-8 -*-


import numpy as np
import cv2
from matplotlib import pyplot as plt


def chromaKey(front, back):
    lower_color = np.array([100/2, 100, 100])
    upper_color = np.array([130/2, 255, 255])

    img_src1 = front
    img_src2 = back

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img_src2, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    inv_mask = cv2.bitwise_not(mask)

    # Bitwise-AND mask,inv_mask and original image
    res1 = cv2.bitwise_and(img_src2,img_src2,mask= inv_mask)
    res2 = cv2.bitwise_and(img_src1,img_src1,mask=  mask)

    #compsiting
    disp = cv2.bitwise_or(res1,res2,mask)
    return disp

# 減色処理
def sub_color(src, K):
    # 次元数を1落とす
    Z = src.reshape((-1,3))
    # float32型に変換
    Z = np.float32(Z)
    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K-means法で減色
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # UINT8に変換
    center = np.uint8(center)
    res = center[label.flatten()]
    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))

def main():
    img = cv2.imread("./images/focus_flower.jpg", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    gray = cv2.medianBlur(gray, ksize=3)

    height, width = gray.shape

    d = 32
    lenH = int(height/d)
    lenW = int(width/d)

    plots = []

    deleteSegments = []

    for i in range(lenH):
        for j in range(lenW):
            idh = i*d
            jdw = j*d
            segImg = gray[idh:idh+d, jdw:jdw+d]
            
            # 高速フーリエ変換(2次元)
            f = np.fft.fft2(segImg)
            # 零周波数成分を配列の左上から中心に移動
            fshift =  np.fft.fftshift(f)
            mgni = 20*np.log(np.abs(fshift))

            # print(i, j)

            plots = []

            for f in range(int(d/2)-5, int(d/2)):
                x0 = y0 = int(d/2)-1-f
                x1 = y1 = int(d/2)+f

                powers = np.hstack( (
                    mgni[y0, x0+1:x1], 
                    mgni[y1, x0+1:x1], 
                    mgni[y0+1:y1, x0], 
                    mgni[y0+1:y1, x1], 
                    mgni[y0, x0], 
                    mgni[y0, x1], 
                    mgni[y1, x0], 
                    mgni[y1, x1]
                    ) )
                powers = list(filter(lambda x: x!=np.inf and x!=-np.inf, powers))
                powers = list(map(lambda x: x*x, powers))
                sumPower = sum(powers)
                rootResult = sumPower/len(powers)
                plots.append(rootResult)

            plotsX = list(map(lambda x:x, range(len(plots))))
            angle, section = np.polyfit(plotsX, plots, 1)
            # angleArr.append(angle)
            
            lsm = np.poly1d(np.polyfit(plotsX, plots, 1))(plotsX)

            # print(angle, section)
            if not(angle<-500 or section<8000):            
                deleteSegments.append([i, j])

            # plt.subplot(lenH, lenW, jdw+idh*lenW+1) # for segments

            # plt.plot(plotsX, lsm, label='d=1') # for lsm
            # plt.plot(plots)

            # plt.imshow(segImg, cmap = 'gray')
            # plt.imshow(mgni, cmap = 'gray')
            # plt.imshow(f, cmap = 'gray')

            # plt.title("%s %s" % (j, i))
            # plt.xticks([])
            # plt.yticks([])
            
    subImg = img.copy()
    subImg = sub_color(subImg, K=4)

    for sgm in deleteSegments:
        i, j = sgm
        height, width, channel = subImg.shape
        dh = int(d*height/256)
        dw = int(d*width/256)
        idh = i*dh
        jdw = j*dw

        cv2.rectangle(subImg, (jdw,idh), (jdw+dw,idh+dh), (0, 255, 0), thickness=-1)

    chromaImg = chromaKey(img, subImg)
    chromaImg = cv2.cvtColor(chromaImg, cv2.COLOR_BGR2RGB)

    plt.subplot()
    plt.imshow(chromaImg)
    
    plt.show()

if __name__ == "__main__":
    main()