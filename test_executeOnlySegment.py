# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("./images/focus_flower.jpg", 1)
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (256, 256))
    height, width = gray.shape

    d = 32
    lenH = int(height/d)
    lenW = int(width/d)

    plots = []

    # for i in range(lenH):
    #     for j in range(lenW):
    i0 = 7*d
    j0 = 3*d
    segImg = gray[i0:i0+d, j0:j0+d]

    plt.subplot(131)
    plt.imshow(segImg, cmap = 'gray')
    # plt.hist(plots)
    plt.title("%s %s" % (4, 7))
    plt.xticks([])
    plt.yticks([])
    
    print(segImg)
    print()

    # 高速フーリエ変換(2次元)
    f = np.fft.fft2(segImg)

    print( f)
    print()

    # 零周波数成分を配列の左上から中心に移動
    fshift =  np.fft.fftshift(f)
    print( fshift)

    # mgni = fshift
    mgni = 20*np.log(np.abs(fshift))

    plt.subplot(132)
    plt.imshow(mgni, cmap = 'gray')
    # plt.hist(plots)
    plt.title("%s %s" % (4, 7))
    plt.xticks([])
    plt.yticks([])

    # print( np.inf in mgni or -np.inf in mgni)
    # print( np.where(mgni is np.inf or mgni is -np.inf) )
    # print( mgni.index(-np.inf))

    plots = []
    for f in range(16):
        x0 = y0 = 15-f
        x1 = y1 = 16+f

        
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

                


    # if i==7 and j==4:
    # plt.subplot(lenH, lenW, 4+7*lenW+1)
    plt.subplot(133)
    # plt.imshow(plots, cmap = 'gray')
    plt.hist(plots)
    plt.title("%s %s" % (4, 7))
    # plt.xticks([])
    # plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()