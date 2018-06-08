import cv2

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.m4v', int(fourcc), 10, (640, 360))

for i in range(623):
    if i<10:
        name = "00%s" % i
    elif i<100:
        name = "0%s" % i
    else:
        name = "%s" % i
    print(name)
    img = cv2.imread('output_imgs/img%s.jpg' % name)
    print(img)
    # img = cv2.resize(img, (640,360))
    video.write(img)

# video.release()