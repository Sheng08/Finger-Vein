# -*- coding: utf-8 -*-
import os
import argparse
import cv2


def get_imgs_roi(save_path, train_data, test_data):

    images = os.listdir(save_path)
    for i, image in enumerate(images):
        # print(i)
        # print("dst_img: ", image)
        img_raw = cv2.imread(os.path.join(save_path, image), 0)
        org_file_root = image.split("_")
        if (i+1)%2 == 0:
            if not os.path.isdir(test_data+org_file_root[0]):
                os.makedirs(test_data+org_file_root[0])
            new_file = test_data+org_file_root[0]+"/"+image
            print("To: ", new_file)
            resized_img = cv2.resize(img_raw, (100, 300), cv2.INTER_LINEAR)
            cv2.imwrite(new_file, resized_img)
        else:
            if not os.path.isdir(train_data+org_file_root[0]):
                os.makedirs(train_data+org_file_root[0])
            new_file = train_data+org_file_root[0]+"/"+image
            print("To: ", new_file)
            resized_img = cv2.resize(img_raw, (100, 300), cv2.INTER_LINEAR)
            cv2.imwrite(new_file, resized_img)

    print("Done...")


if __name__ == '__main__':
    print("會覆蓋原始ROI Result檔案")
    print("請輸入目標離散小波資料夾(圖片為單層資料夾)")
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--save_path",
                        dest="save_path",
                        help="input save path e.g. ./ROI_Result/",
                        type=str)
    parser.add_argument("-r",
                        "--train_data",
                        dest="train_data",
                        help="input train_data path e.g. ./train_data/",
                        type=str)
    parser.add_argument("-t",
                        "--test_data",
                        dest="test_data",
                        help="input test_data path e.g. ./test_data/",
                        type=str)
    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        print("Not match folder!!")
    else:
        get_imgs_roi(args.save_path, args.train_data, args.test_data)
