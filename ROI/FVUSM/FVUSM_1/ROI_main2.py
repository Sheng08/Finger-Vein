# -*- coding: utf-8 -*-
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy
import os

import imutils
import argparse
import pywt


def first_filter(img):
    # 均值滤波
    # img_Blur = cv2.blur(img, (3, 3))
    # 高斯滤波
    # img_GaussianBlur = cv2.GaussianBlur(img, (3, 3), 0)
    # 高斯双边滤波
    # img_bilateralFilter = cv2.bilateralFilter(img, 17, 15, 1)
    img_bilateralFilter = cv2.bilateralFilter(img, 19, 3, 210)
    return img, img_bilateralFilter

########################
# 边缘检测


def edge_detection(img):
    # img = cv2.imread(file, 0)
    # img = cv2.imread("01.jpg", 0)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    img_edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 7)
    # 與原圖相同亮度 所以在做相減時 與原圖相同元素砍掉 只留下邊邊亮的地方 且因為邊邊比原本亮 所以與原圖相減後是呈現最暗
    img_edge = cv2.addWeighted(img, 0.5, img_edge, 0.5, 0)
    # img_edge = cv2.subtract(img, img_edge)
    img_edge = cv2.add(img, img_edge)
    canny_th3 = cv2.Canny(img_edge, 30, 120)

    # 用哪種閉運算好?
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # MORPH_CLOSE = cv2.morphologyEx(canny_th3, cv2.MORPH_CLOSE, kernel)  # 降黑點
    MORPH_CLOSE = cv2.dilate(canny_th3, kernel, iterations=3)
    final_edge = cv2.erode(MORPH_CLOSE, kernel, iterations=2)

    fig = plt.figure(figsize=(16, 16))
    fig.canvas.set_window_title('edge_detection')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.title.set_text('Original')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax2.title.set_text('img_edge')
    ax2.imshow(img_edge, cmap=plt.cm.gray)
    ax3.title.set_text('canny_th3')
    ax3.imshow(canny_th3, cmap=plt.cm.gray)
    ax4.title.set_text('final_edge')
    ax4.imshow(final_edge, cmap=plt.cm.gray)
    plt.show()

    return img, final_edge

############################################
# 像素二值化


def pixel_polarization(img_edge, img, threshold):  # threshold 像素两极化的阈值
    for i in range(len(img_edge)):
        for j in range(len(img_edge[i, :])):
            if img_edge[i][j] > threshold:
                img_edge[i][j] = 255
            else:
                img_edge[i][j] = 0
    img_edge_polar = img_edge
    return img_edge_polar


def positioning_middle_point(img, dst, point_pixel):
    h, w = img.shape
    w1 = w // 5  # 作为左边竖线的x坐标
    w2 = (w // 5) * 4  # 作为右边竖线的x坐标

    # print("roi width: ", h, w1, w2)

    low_l = False
    high_l = False
    while (not low_l or not high_l) and w1 < (w // 2):
        for i, pix in enumerate(dst[:, w1]):
            if i+1 < (h // 2) and not low_l:
                h_h = int(h * (1/2) - (i+1))
                if dst[h_h, w1] == 255:
                    low_l = True
                    lower_left = h_h
            elif i+1 > (h // 2) and not high_l:
                # 除法会带来小数，因此用int(), h/2开始对称位置找亮点
                if pix == 255:
                    high_l = True
                    higher_left = i
        if (not low_l or not high_l):
            w1 = w1 + 2
        elif (abs(higher_left-lower_left) < 100):
            w1 = w1 + 2
            low_l = False
            high_l = False

    low_r = False
    high_r = False
    while (not low_r or not high_r) and w2 > (w // 2):
        for i, pix in enumerate(dst[:, w2]):
            if i+1 < (h // 2) and not low_r:
                h_h = int(h * (1/2) - (i+1))
                if dst[h_h, w2] == 255:
                    low_r = True
                    lower_right = h_h
            elif i+1 > (h // 2) and not high_r:

                if pix == 255:
                    high_r = True
                    higher_right = i
        if (not low_r or not high_r):
            w2 = w2 - 2
        elif (abs(higher_right-lower_right) < 100):
            w2 = w2 - 2
            low_r = False
            high_r = False
    middle_left = (lower_left + higher_left) // 2
    middle_right = (lower_right + higher_right) // 2
    # try:
    #     middle_left = (lower_left + higher_left) // 2
    #     middle_right = (lower_right + higher_right) // 2
    # except Exception as e:
    #     print(e)
    #     fig = plt.figure(figsize=(16, 16))
    #     fig.canvas.set_window_title('positioning_middle_point_Exception')
    #     ax1 = fig.add_subplot(1, 1, 1)
    #     ax1.title.set_text('Exception_dst')
    #     ax1.imshow(dst, cmap=plt.cm.gray)
    #     plt.show()
    # print("lower_left", lower_left)

    # dst[middle_left, w1] = point_pixel
    # dst[middle_left+1, w1] = point_pixel
    # dst[middle_left-1, w1] = point_pixel
    # dst[middle_left, w1 + 1] = point_pixel
    # dst[middle_left, w1 - 1] = point_pixel
    # dst[middle_right, w2] = point_pixel
    # dst[middle_right+1, w2] = point_pixel
    # dst[middle_right-1, w2] = point_pixel
    # dst[middle_right, w2 + 1] = point_pixel
    # dst[middle_right, w2 - 1] = point_pixel

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('positioning_middle_point')
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.title.set_text('Original')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax2.title.set_text('middle_point')
    # ax2.imshow(dst, cmap=plt.cm.gray)
    # plt.show()

    return dst, middle_left, middle_right, w1, w2


def rotated_minWidth(img, dst, w1, w2, point_pixel):
    h, w = img.shape

    # print("roi width: ", h, w1, w2)

    # 確定左右邊界一定會和邊界上的邊緣相交
    # TODO 做例外處理或警告提示 改成以右邊為判斷 因為手指的型態規則
    '''
    low_l = False
    high_l = False
    while (not low_l or not high_l) and w1 < (w // 2):
        for i, pix in enumerate(dst[:, w1]):
            if i+1 < (h // 2) and not low_l:
                h_h = int(h * (1/2) - (i+1))
                if dst[h_h, w1] == 255:
                    low_l = True
                    lower_left = h_h
            elif i+1 > (h // 2) and not high_l:
                if pix == 255:
                    high_l = True
                    higher_left = i
        if (abs(higher_left-lower_left) < 100):
            w1 = w1 + 2
            low_l = False
            high_l = False

    middle_left = (lower_left + higher_left) // 2
    width_left = abs(higher_left - lower_left)
    lower_left_length = abs((h // 2)-lower_left)
    higher_left_length = abs((h // 2)-higher_left)
    '''
    low_r = False
    high_r = False
    while (not low_r or not high_r) and w2 > (w // 2):
        for i, pix in enumerate(dst[:, w2]):
            if i+1 < (h // 2) and not low_r:
                h_h = int(h * (1/2) - (i+1))
                if dst[h_h, w2] == 255:
                    low_r = True
                    lower_right = h_h
            elif i+1 > (h // 2) and not high_r:
                if pix == 255:
                    high_r = True
                    higher_right = i
        if (not low_r or not high_r):
            w2 = w2 - 2
        elif (abs(higher_right-lower_right) < 100):
            # print("AA", abs((h // 2)-lower_right))
            # print("BB", abs((h // 2)-higher_right))
            w2 = w2 - 2
            low_r = False
            high_r = False
        # try:
        #     if (abs(higher_right-lower_right) < 100):
        #         # print("AA", abs((h // 2)-lower_right))
        #         # print("BB", abs((h // 2)-higher_right))
        #         w2 = w2 - 2
        #         low_r = False
        #         high_r = False
        # except Exception as e:
        #     print(e)
        #     middle_right = (lower_right + higher_right) // 2
        #     dst[middle_right, w2] = point_pixel
        #     dst[middle_right+1, w2] = point_pixel
        #     dst[middle_right-1, w2] = point_pixel
        #     dst[middle_right, w2 + 1] = point_pixel
        #     dst[middle_right, w2 - 1] = point_pixel
        #     fig = plt.figure(figsize=(16, 16))
        #     fig.canvas.set_window_title('Exception')
        #     ax1 = fig.add_subplot(1, 1, 1)
        #     ax1.title.set_text('Exception_dst')
        #     ax1.imshow(dst, cmap=plt.cm.gray)
        #     plt.show()

    middle_right = (lower_right + higher_right) // 2
    # width_right = abs(higher_right - lower_right)
    lower_right_length = abs((h // 2)-lower_right)
    higher_right_length = abs((h // 2)-higher_right)
    width_lower = lower_right_length
    width_higher = higher_right_length
    # width_lower = lower_left_length if lower_left_length < lower_right_length else lower_right_length
    # width_higher = higher_left_length if higher_left_length < higher_right_length else higher_right_length

    # print("lower_left_length", lower_left_length)
    # print("lower_right_length", lower_right_length)
    # print("higher_left_length", higher_left_length)
    # print("higher_right_length", higher_right_length)
    # print("rotated_minWidth_lower_left", lower_left)
    # print("rotated_minWidth_higher_left", higher_left)
    # print("rotated_minWidth_lower_right", lower_right)
    # print("rotated_minWidth_higher_right", higher_right)

    # dst[middle_left, w1] = point_pixel
    # dst[middle_left+1, w1] = point_pixel
    # dst[middle_left-1, w1] = point_pixel
    # dst[middle_left, w1 + 1] = point_pixel
    # dst[middle_left, w1 - 1] = point_pixel
    # dst[middle_right, w2] = point_pixel
    # dst[middle_right+1, w2] = point_pixel
    # dst[middle_right-1, w2] = point_pixel
    # dst[middle_right, w2 + 1] = point_pixel
    # dst[middle_right, w2 - 1] = point_pixel

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('rotated_minWidth')
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.title.set_text('Original')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax2.title.set_text('middle_point')
    # ax2.imshow(dst, cmap=plt.cm.gray)
    # plt.show()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.title.set_text('Original')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax2.title.set_text('rotated_minWidth')
    # ax2.imshow(dst, cmap=plt.cm.gray)
    # plt.show()

    return width_lower, width_higher
#################################
# 旋转矫正


def rotation_correction(img, dst, middle_right, middle_left, w1, w2):
    tangent_value = float(middle_right - middle_left) / float(w2 - w1)
    rotation_angle = np.arctan(tangent_value) / \
        math.pi*180  # 弧度pi轉角度 比例運算 tan()裡面放弧度
    # print("rotation_angle", rotation_angle)
    (h, w) = img.shape
    center = (w // 2, h // 2)

    # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    M = cv2.getRotationMatrix2D(center, -rotation_angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    rotated_dst = cv2.warpAffine(dst, M, (w, h))

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('rotation_correction')
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.title.set_text('Original')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax2.title.set_text('rotated_img')
    # ax2.imshow(rotated_img, cmap=plt.cm.gray)
    # ax3.title.set_text('rotated_dst')
    # ax3.imshow(rotated_dst, cmap=plt.cm.gray)
    # plt.show()

    return rotated_img, rotated_dst


def roi(rotated_img, rotated_edge, w1, w2, width_lower, width_higher, roi_result_root):
    h, w = rotated_edge.shape
    r = range(0, h)
    r1 = range(0, h // 2)
    r2 = range(h // 2, h - 1)
    c = range(0, w)
    c1 = range(0, w // 2)
    c2 = range(w // 2, w-1)

    # print("width_lower", width_lower)
    # print("width_higher", width_higher)

    lowest_edge = (h//2)-width_lower
    highest_edge = (h//2)+width_higher
    # print("highest_edge", highest_edge)
    # print("lowest_edge", lowest_edge)

    leftest_edge = w1
    rightest_edge = w2

    rotated_croped_img = rotated_img[lowest_edge: highest_edge,
                                     leftest_edge: rightest_edge]

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('roi')
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.title.set_text('rotated_img')
    # ax1.imshow(rotated_img, cmap=plt.cm.gray)
    # ax2.title.set_text('rotated_edge')
    # ax2.imshow(rotated_edge, cmap=plt.cm.gray)
    # ax3.title.set_text('ROI Result')
    # ax3.imshow(rotated_croped_img, cmap=plt.cm.gray)
    # plt.show()

    # print("rotated_croped_img type: ", rotated_croped_img)
    # cv2.imwrite(url, rotated_croped_img)

    # im = Image.fromarray(rotated_croped_img)
    # im.save(url)

    #####draw_rectangle#####
    draw_rectangle_img = rotated_img.copy()
    draw_rectangle_img_RGB = cv2.cvtColor(
        draw_rectangle_img, cv2.COLOR_GRAY2BGR)
    left_up = (leftest_edge, lowest_edge)
    right_down = (rightest_edge, highest_edge)
    color = (0, 255, 0)  # red
    thickness = 1  # 寬度 (-1 表示填滿)
    cv2.rectangle(draw_rectangle_img_RGB, left_up,
                  right_down, color, thickness)

    draw_rectangle_img_RGB = cv2.transpose(draw_rectangle_img_RGB)
    draw_rectangle_img_RGB = cv2.flip(draw_rectangle_img_RGB, 1)  # 順時鐘90
    cv2.imwrite(roi_result_root, draw_rectangle_img_RGB)

    return rotated_croped_img


def DWT2_cA(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('haar')
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    # ax1.title.set_text('A')
    # ax1.imshow(cA, cmap=plt.cm.gray)
    # ax2.title.set_text('H')
    # ax2.imshow(cH, cmap=plt.cm.gray)
    # ax3.title.set_text('V')
    # ax3.imshow(cV, cmap=plt.cm.gray)
    # ax4.title.set_text('D')
    # ax4.imshow(cD, cmap=plt.cm.gray)
    # plt.show()

    return cA


def img_resized_enhance(img, url):
    # 小波轉換
    
    # 先小小波再resize 解析度比較佳
    img = DWT2_cA(img)
    # 尺度归一化
    # resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    resized_img = cv2.resize(img, (300, 100), cv2.INTER_LINEAR)  # 双线性插值
    # resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    '''
    fig = plt.figure(figsize = (30, 20))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(resized_img, cmap = plt.cm.gray)
    plt.show()
    '''
    # norm_resized_img = resized_img.copy()
    # 灰度归一化
    # norm_resized_img = cv2.normalize(
    #     resized_img, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    # 直方图均衡化
    # equ_resized_img = cv2.equalizeHist(resized_img)
    #create a CLAHE object (Arguments are optional)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #resized_img = clahe.apply(resized_img)

    #######DWT#######
    # 先resize在小波
    # resized_img = DWT2_cA(resized_img)

    # fig = plt.figure(figsize=(16, 16))
    # fig.canvas.set_window_title('img_resized_enhance')
    # ax1 = fig.add_subplot(1, 1, 1)
    # # ax2 = fig.add_subplot(1, 3, 2)
    # # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.title.set_text('resized_img')
    # ax1.imshow(resized_img, cmap=plt.cm.gray)
    # # ax2.title.set_text('norm_resized_img')
    # # ax2.imshow(norm_resized_img, cmap=plt.cm.gray)
    # # ax3.title.set_text('clahe_resized_img')
    # # ax3.imshow(clahe_resized_img, cmap=plt.cm.gray)
    # plt.show()

    #######resize#######
    resized_img = cv2.transpose(resized_img)
    resized_img = cv2.flip(resized_img, 1)  # 順時鐘90
    cv2.imwrite(url, resized_img)

    #######normalize#######
    # norm_resized_img = cv2.transpose(norm_resized_img)
    # norm_resized_img = cv2.flip(norm_resized_img, 1)  # 順時鐘90
    # cv2.imwrite(url, norm_resized_img)

    #######normalize+clahe_resized_img#######
    # clahe_resized_img = cv2.transpose(clahe_resized_img)
    # clahe_resized_img = cv2.flip(clahe_resized_img, 1)  # 順時鐘90
    # cv2.imwrite(url, clahe_resized_img)

    return resized_img


def get_imgs_roi(img_file, save_path, expcetion_path, roi_result_path):
    count = 0
    all_folder = os.listdir(img_file)
    # print(all_folder)
    # os.makedirs("./expcetion_dst")
    if not os.path.isdir("./expcetion_dst"):
        os.makedirs("./expcetion_dst")
    for _, images in enumerate(all_folder):
        if not os.path.isdir(roi_result_path+"draw_"+images):
            os.makedirs(roi_result_path+"draw_"+images)
            
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        if not os.path.isdir(expcetion_path):
            os.makedirs(expcetion_path)
            
        for i, image in enumerate(os.listdir(img_file+images)):
            # print(i)
            # print(image)
            img_raw = cv2.imread(os.path.join(img_file+images+'/', image), 0)
            print(img_file+images+'/'+image)
            # print(img_raw.shape)

            # (h, w) = img_raw.shape
            # center = (w / 2, h / 2)
            # # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
            # M = cv2.getRotationMatrix2D(center, 90, 1)
            # img_raw = cv2.warpAffine(img_raw, M, (w, h))
            img_raw = cv2.transpose(img_raw)
            img_raw = cv2.flip(img_raw, 0)  # 逆時鐘90

            img_raw, img_Blur = first_filter(img_raw)
            img_raw, img_Blur_edge = edge_detection(img_Blur)

            # img_Blur_edge_polar = pixel_polarization(
            #     img_Blur_edge, img_raw, 25)  # 二值化
            try:
                img_Blur_edge_polar_midd, middle_left, middle_right, w1, w2 = positioning_middle_point(
                    img_raw, img_Blur_edge, 150)

                rotated_img, img_Blur_edge_polar_midd_rotated = rotation_correction(
                    img_raw, img_Blur_edge_polar_midd,  middle_left, middle_right, w1, w2)

                width_lower, width_higher = rotated_minWidth(
                    rotated_img, img_Blur_edge_polar_midd_rotated, w1, w2,  255)

                # roi图像保存路径
                images1 = images.split("_")
                img = image.split(".")
                # print(int(img[0]))
                new_file = save_path + \
                    str(int(images1[0]))+"_"+str(int(images1[1])) + \
                    "_"+str(int(img[0]))+"."+img[1]

                print(new_file)
                save_root = os.path.join(new_file, image)

                new_file2 = roi_result_path + 'draw_'+images
                roi_result_root = os.path.join(new_file2, image)
                roi_img = roi(
                    rotated_img, img_Blur_edge_polar_midd_rotated, w1, w2, width_lower, width_higher, roi_result_root)
                resized_roi_img = img_resized_enhance(roi_img, new_file)
            except Exception as e:
                print(e)
                count += 1
                # fig = plt.figure(figsize=(16, 16))
                # fig.canvas.set_window_title('Exception')
                # ax1 = fig.add_subplot(1, 1, 1)
                # ax1.title.set_text('Exception_dst')
                # ax1.imshow(img_Blur_edge_polar, cmap=plt.cm.gray)
                # plt.show()

                img_raw = cv2.imread(os.path.join(
                    img_file+images+'/', image), 0)
                _file = expcetion_path+images+"_"+image
                # save_root = os.path.join(new_file, image)
                cv2.imwrite(_file, img_raw)
                # _file2 = expcetion_path+"dst/"+"dst_"+images+"-"+image
                _file2 = "./expcetion_dst/"+"dst_"+images+"_"+image
                # save_root = os.path.join(new_file, image)
                cv2.imwrite(_file2, img_Blur_edge)
    print("Done...")
    print("Exception:", count)


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        params = {'ksize': (ksize, ksize), 'sigma': 3.3, 'theta': theta, 'lambd': 18.3,
                  'gamma': 4.5, 'psi': 0.89, 'ktype': cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern, params))
    return filters


def getGabor(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


################################################################
# 二值纹理特征提取
def bin_features_extract(roi_file):
    '''
    images_roi = os.listdir(roi_file)
    for i, image_roi in enumerate(images_roi):
        print(i)
        print(image_roi)
        img_roi_raw = cv2.imread(os.path.join(roi_file, image_roi), 0)
    '''
    img_roi_raw = cv2.imread(roi_file, 0)
    # Gabor滤波器
    filters = build_filters()
    img_roi_raw_gabor = getGabor(img_roi_raw, filters)
    # print(img_roi_raw_gabor)
    # 灰度归一化
    # norm_resized_img = cv2.normalize(img_roi_raw_gabor, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    # 二值化
    # img_roi_raw_gabor_polar60 = img_roi_raw_gabor.copy()
    # img_roi_raw_gabor_polar60 = pixel_polarization(img_roi_raw_gabor_polar60, img_roi_raw, 60)
    img_roi_raw_gabor_polar70 = img_roi_raw_gabor.copy()
    img_roi_raw_gabor_polar70 = pixel_polarization(
        img_roi_raw_gabor_polar70, img_roi_raw, 70)
    '''
        plt.figure(figsize = (30, 30))
        plt.subplot(2, 2, 1), plt.title('img_roi_raw')
        plt.imshow(img_roi_raw, cmap = plt.cm.gray)
        plt.subplot(2, 2, 2), plt.title('img_roi_raw_gabor')
        plt.imshow(img_roi_raw_gabor, cmap = plt.cm.gray)
        plt.subplot(2, 2, 3), plt.title('img_roi_raw_gabor_polar60')
        plt.imshow(img_roi_raw_gabor_polar60, cmap = plt.cm.gray)
        plt.subplot(2, 2, 4), plt.title('img_roi_raw_gabor_polar70')
        plt.imshow(img_roi_raw_gabor_polar70, cmap = plt.cm.gray)
        plt.show()
    '''

    return img_roi_raw_gabor_polar70


def bin_match(img1_path, img2_path):
    img1 = bin_features_extract(img1_path)
    img2 = bin_features_extract(img2_path)
    height, width = img1.shape
    size = height * width
    score = 0
    for i in range(len(img1)):
        for j in range(len(img1[i, :])):
            if img1[i][j] == img2[i][j]:
                score += 1
    scores = 100 * round((score / size), 4)
    # print(img1_path, img2_path, scores)
    return scores


###########################################################
# 图片分成m*n块
def cut_image(image, m, n):
    height, width = image.shape
    item_width = int(width // m)
    item_height = int(height // n)
    # box_list = []
    cropped_list = []
    # (left, upper, right, lower)
    for i in range(0, n):  # 两重循环，生成m*n张图片基于原图的位置
        for j in range(0, m):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            # box = (j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height)
            # box_list.append(box)
            cropped = image[i*item_height:(i+1)*item_height,
                            j*item_width:(j+1)*item_width]
            cropped_list.append(cropped)

    print(len(cropped_list))
    # image_list = [image.crop(box) for box in box_list]
    return cropped_list



##########################################
# SIFT特征提取与匹配


def SIFT_detector(gray_path):
    images_sift = os.listdir(gray_path)
    for i, image_sift in enumerate(images_sift):
        print(i)
        print(image_sift)
        img = cv2.imread(os.path.join(gray_path, image_sift), 0)
        '''
        # sift检测
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(img,None)
        img_sift=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        '''
        # SURF检测
        surf = cv2.xfeatures2d.SURF_create()
        kp = surf.detect(img,None)
        img_surf=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        '''
        # ORB检测，几乎没有
        orb = cv2.ORB_create()
        kp = orb.detect(img,None)
        img_orb=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''

        # KAZE检测
        kaze = cv2.KAZE_create()
        kp = kaze.detect(img, None)
        img_kaze = cv2.drawKeypoints(
            img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # cv2.imwrite('sift_keypoints.jpg',img)

        plt.figure(figsize=(30, 30))
        plt.subplot(1, 2, 1), plt.title('img')
        plt.imshow(img, cmap=plt.cm.gray)
        plt.subplot(1, 2, 2), plt.title('img_kaze')
        plt.imshow(img_kaze, cmap=plt.cm.gray)
#        plt.subplot(1, 3, 3), plt.title('lbp_hist')
#        plt.imshow(lbp_hist)
        plt.show()


def SIFT_match(img1_path, img2_path):

    img1 = cv2.imread(img1_path, 0)          # queryImage
    img2 = cv2.imread(img2_path, 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()




if __name__ == '__main__':
    get_imgs_roi("./FVUSM1_raw_data/", "./Result/Harr/",
                    "./Result/Expc/Harr_expc/", "./Result/Draw/Harr_draw/")

    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--folder_path",
                        dest="folder_path",
                        help="input folder path e.g. ./raw_data/",
                        type=str)
    parser.add_argument("-s",
                        "--save_path",
                        dest="save_path",
                        help="input save path e.g. ./ROI_Result/",
                        type=str)
    parser.add_argument("-e",
                        "--expcetion_path",
                        dest="expcetion_path",
                        help="input expcetion path e.g. ./expcetion/",
                        type=str)
    parser.add_argument("-r",
                        "--roi_result_path",
                        dest="roi_result_path",
                        help="input roi_result path e.g. ./roi_result_path/",
                        type=str)
    args = parser.parse_args()
    if not os.path.isdir(args.folder_path):
        print("Not match Data folder!!")
    else:
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.isdir(args.expcetion_path):
            os.mkdir(args.expcetion_path)
        if not os.path.isdir(args.roi_result_path):
            os.mkdir(args.roi_result_path)
        # for folder in args:
        #     if not os.path.isdir(folder):
        #         os.mkdir(folder)
        # get_imgs_roi('./test_rawData/')
        get_imgs_roi(args.folder_path, args.save_path,
                     args.expcetion_path, args.roi_result_path)
        """