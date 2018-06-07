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
    img = cv2.imread("./images/focus_car04.jpg", 1)
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (256, 256))
    # gray = gray0
    gray = median_filter(gray, ksize=3)

    height, width = gray.shape

    d = 32
    lenH = int(height/d)
    lenW = int(width/d)

    plots = []

    # angleArr = []
    deleteSegments = []
    for i in range(lenH):
        for j in range(lenW):
            # if i==7 and j==4: continue

            i0 = i*d
            j0 = j*d
            segImg = gray[i0:i0+d, j0:j0+d]
            
            # 高速フーリエ変換(2次元)
            f = np.fft.fft2(segImg)

            # 零周波数成分を配列の左上から中心に移動
            fshift =  np.fft.fftshift(f)

            mgni = 20*np.log(np.abs(fshift))
            # mgni = np.log(np.abs(fshift))

            # print(i,j,mgni)

            plots = []
            # for f in range(16):
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
                powers = list(map(lambda x: x*x, powers))
                sumPower = sum(powers)
                rootResult = sumPower/len(powers)
                plots.append(rootResult)

            plotsX = list(map(lambda x:x, range(len(plots))))
            angle, section = np.polyfit(plotsX, plots, 1)
            # angleArr.append(angle)
            
            lsm = np.poly1d(np.polyfit(plotsX, plots, 1))(plotsX)

            # print(angle, section)
            if angle<-500 or section<8000:            
                deleteSegments.append([i, j])

                plt.subplot(lenH, lenW, j+i*lenW+1)

                # plt.plot(plotsX, lsm, label='d=1')
                
                # if i==7 and j==4:
                # plt.subplot(111)

                # plt.imshow(segImg, cmap = 'gray')
                # plt.imshow(mgni, cmap = 'gray')
                # plt.imshow(f, cmap = 'gray')

                # plt.plot(plots)

                # plt.title("%s %s" % (j, i))
                # plt.xticks([])
                # plt.yticks([])

    # print(angleArr)
    for sgm in deleteSegments:
        i, j = sgm
        height, width, channel = img.shape
        dh = int(d*height/256)
        dw = int(d*width/256)
        # dh = dw = 32
        i0 = i*dh
        j0 = j*dw
        # img[ [i0:i0+dh, j0:j0+dw ] = 0
        for p in range(i0, i0+dh):
            for q in range(j0, j0+dw):
                img[p][q] = 0

    plt.subplot(111)
    plt.imshow(img)
    
    plt.show()

if __name__ == "__main__":
    main()