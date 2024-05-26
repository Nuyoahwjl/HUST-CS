# img_show.py
# 注意本程序的运行需要配置好dl环境下的解释器
import cv2 as cv
name = 'java_from_introduction_to_mastery.png' # when_i_was_young.png, java_from_introduction_to_mastery.png
img = cv.imread("img/" + name)
img_d = img[:,:,-1]
img_c = img[:,:,2]
# cv.imshow(name, img_d)
cv.imshow(name, img_c)
cv.waitKey(4000) # 显示4秒