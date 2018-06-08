# -*- coding: utf-8 -*-


import numpy as np
import cv2
from matplotlib import pyplot as plt

# flag : 0...camera capture mode / 1...image mode
flag = 0
images = []
cnt = 0


def chromaKey(front, back, lc, uc):
    lower_color = np.array(lc)
    upper_color = np.array(uc)

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
 
    if flag==0:
        capture = cv2.VideoCapture('./images/video02.mov')
        # capture = cv2.VideoCapture(0)

        # capture.set(3, 320)
        # capture.set(4, 240)
        # capture.set(5, 30)


        while True:
            _, img = capture.read()
            process(img)

            if cv2.waitKey(25) > 0:
                break
        # images[0].save(
        #     'images/exportGif.gif',
        #     save_all=True, 
        #     append_images=images[1:], 
        #     optimize=False, 
        #     duration=40, 
        #     loop=0)
        capture.release()
        cv2.destroyAllWindows()
    
    elif flag==1:
        process([])

def process(img):
    if flag==1: img = cv2.imread("./images/focus_car01.jpg")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.Laplacian(gray, cv2.CV_32F)
    gray = cv2.convertScaleAbs(gray)
    # _, gray = cv2.threshold(gray, 30, 50, cv2.THRESH_BINARY)

    height, width = gray.shape
    d = 8
    lenH = int(height/d)
    lenW = int(width/d)

    blackSeg = []
    graySeg  = []
    for h in range(lenH):
        for w in range(lenW):
            dh = h*d
            dw = w*d
            segImg = gray[dh:dh+d, dw:dw+d]
            ave = np.mean(segImg)
            flat = 0
            if ave<9:
                flat = 0
                blackSeg.append([h,w])
            elif ave<46:
                flat = 50
                graySeg.append([h,w])
            elif ave<256:
                flat = 255

            cv2.rectangle(gray, (dw,dh), (dw+d,dh+d), flat, thickness=-1)
                
    blackImg = img.copy()
    blackImg = sub_color(blackImg, K=8)

    grayImg = img.copy()
    grayImg = sub_color(grayImg, K=16)

    for sgm in blackSeg:
        h, w = sgm
        height, width, channel = blackImg.shape
        dh = h*d
        dw = w*d
        cv2.rectangle(img, (dw,dh), (dw+d,dh+d), (0,255,0), thickness=-1)

    for sgm in graySeg:
        h, w = sgm
        height, width, channel = grayImg.shape
        dh = h*d
        dw = w*d
        cv2.rectangle(img, (dw,dh), (dw+d,dh+d), (255,0,0), thickness=-1)

    chromaImg = chromaKey(blackImg, img, lc=[100/2, 100, 100], uc=[130/2, 255, 255])
    chromaImg2 = chromaKey(grayImg, chromaImg, lc=[230/2, 100, 100], uc=[250/2, 255, 255])
   
    # print(len(images))
    # images.append(chromaImg2)
    cv2.imshow('Dot Focus', chromaImg2)

    global cnt
    if cnt<10:
        num = "00%s" % cnt
    elif cnt<100:
        num = "0%s" % cnt
    else:
        num = cnt
    cnt+=1

    
    cv2.imwrite("img%s.jpg"%(num), chromaImg2)

    if flag==1:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()