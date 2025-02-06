import datetime
import random
import shutil
import numpy as np
import base64
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import qrcode

import os
from pathlib import Path
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

chunk_size = 30 * 30
outPath = 'outPut'
qrMaxSizeDic = {
    qrcode.constants.ERROR_CORRECT_L: '2950',
    qrcode.constants.ERROR_CORRECT_M: '2330',
    qrcode.constants.ERROR_CORRECT_Q: '1660',
    qrcode.constants.ERROR_CORRECT_H: '1270'

}



def print_qr(_base64_str_list):
    # 循环生成二维码图片
    error_correction_1 = qrcode.constants.ERROR_CORRECT_M

    imgNums = 0
    startSize = 0
    maxSize = 800 - 3  # 留三位用于编码位置顺序
    endSize = len(_base64_str_list)
    imgsize = 8  # 每张图像中存放二维码的数量
    image_list = []
    while True:
        if startSize >= endSize:
            break
        if (imgNums <= 7):
            imgindex = 0
        else:
            imgindex = (imgNums - 8) // imgsize % 999 + 1  # 一张图的八张二维码编码一致，1000张图之后重新清零

        _in_str_list = "{:03}".format(imgindex)
        if startSize + maxSize + 3 > endSize:
            lensize = startSize + maxSize + 3 - endSize
            _in_str_list += _base64_str_list[startSize:endSize]
            for t in range(lensize):
                _in_str_list += "="
        else:

            _in_str_list += _base64_str_list[startSize:maxSize + startSize]
        qr = qrcode.QRCode(version=10,
                           error_correction=error_correction_1,
                           border=4,
                           box_size=10)

        qr.add_data(_in_str_list)
        img = qr.make_image()  # 2维
        # img = qr.make_image(fill_color="rgb(242,123,0)", back_color="white")
        image_list.append(img)
        imgNums = imgNums + 1
        startSize = startSize + maxSize
        print(f"第{imgNums}张二维码已生成")
    return image_list


def create_qrCode(filePath):
    _, filetype = filePath.split(".")
    filetype = filetype.zfill(4)  # 将文件类型填充到4位

    def read_in_chunks(file_object, chunk_size=1024):
        while True:
            data = file_object.read(chunk_size)
            if not data:
                break
            yield data

    with open(filePath, "rb") as file:
        data = filetype + base64.b64encode(b''.join(read_in_chunks(file))).decode("utf-8")

    imagelist = print_qr(data)

    return imagelist


def pingjie(imagelist):
    # 大图片尺寸
    big_image_size = (3840, 2160)
    # 每块大小
    block_width = big_image_size[0] // 4
    block_height = big_image_size[1] // 2
    block_size = (block_width, block_height)
    # 小二维码尺寸
    small_qr_size = (800, 800)
    image_number = 8  # 一张图填充个数
    # 创建紫色填充图像
    purple_color = (128, 0, 128)
    # purple_color = (66, 172, 176) # 66 172 176
    # 遍历文件夹内的图片
    num = len(imagelist)
    imagelist_8 = []

    count = 0
    for i in range(0, num // 8 + 1):
        # 创建大图片
        big_image = Image.new('RGB', big_image_size, purple_color)

        for j in range(8):
            if (i * 8 + j + 1) > num:
                break
            # 打开小二维码
            small_qr = imagelist[i * 8 + j]
            # 缩放小二维码
            small_qr = small_qr.resize(small_qr_size)

            # 计算放置位置
            left = (j % 4) * block_width
            top = (j // 4) * block_height

            # 计算二维码在紫色底片中的居中位置
            x_offset = (block_width - small_qr_size[0]) // 2
            y_offset = (block_height - small_qr_size[1]) // 2

            # 将二维码居中放置在紫色底片中
            big_image.paste(small_qr, (left + x_offset, top + y_offset))

        # 保存带二维码的大图片到文件夹B
        imagelist_8.append(big_image)
        count += 1
    return imagelist_8


def img2vedio(image_list, video_path):
    # 图像所在文件夹路径
    # 输出视频的路径
    video_path = video_path + '/output_video.avi'
    width, height = image_list[0].size

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 使用DIVX编码器
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    new_imagelist = []
    for image in image_list:
        new_imagelist.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    for i in range(10):
        video.write(new_imagelist[0])
    for image in new_imagelist[:]:
        video.write(image)
    for i in range(10):
        video.write(new_imagelist[0])
    for image in new_imagelist[:]:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
