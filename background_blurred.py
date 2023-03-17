# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils                    # mediapipe 繪圖功能
mp_selfie_segmentation = mp.solutions.selfie_segmentation  # mediapipe 自拍分割方法

kernel_size=(7,7)
current_pos = None
tl = None
br = None

alpha_slider_max = 100
title_window = 'Linear Blend'
def on_trackbar(val):
    alpha = val / alpha_slider_max*2.5
    alpha=2.5-alpha
    sigma = ( 3 - alpha )
    print("kernel_size",kernel_size)
    #dst = cv2.addWeighted(img, alpha, img, beta, 0.0)#模糊化寫這邊
    dst = cv2.GaussianBlur(img, kernel_size, sigma)
    background_dst = cv2.GaussianBlur(image, kernel_size, sigma)
    with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
    #img = cv2.resize(bg,(520,300))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = selfie_segmentation.process(img2)   # 取得自拍分割結果
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1 # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
        output_image = np.where(condition, img, dst)
        
        rows =  img.shape[0]
        cols=img.shape[1]
        for w in  range(rows):
            for h in range(cols):
                background_dst[(x1+w),(y1+h)]=output_image[w,h]
        cv2.imshow(title_window, background_dst)
#鼠标事件
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))
    
    return tl[0],tl[1],br[0],br[1]  #tl=(y1,x1), br=(y2,x2)


image=cv2.imread("C:\\Users\\liu\\Pictures\\article-5bfddfd3682a4.jpg")
#image=cv2.resize(image,(500,500))
y1,x1,y2,x2=get_rect(image, title='get_rect')

img=image[x1:x2,y1:y2]
#cv2.imshow("imm",img)
cv2.namedWindow(title_window)
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
# Show some stuff


if(math.floor(image.shape[1]/67)%2==0):
    kernel_size=(math.floor(image.shape[1]/67)+1,math.floor(image.shape[1]/67)+1)
else:
    kernel_size=(math.floor(image.shape[1]/67),math.floor(image.shape[1]/67))
print("kernel_size",kernel_size)
on_trackbar(0)
# Wait until user press some key
cv2.waitKey()
