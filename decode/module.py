# -*- coding: utf-8 -*-
import time
from tempfile import TemporaryDirectory

import cv2
#pip install Pillow pyzbar
import os
from PIL import Image
from cimbar.cimbar import decode
import numpy as np
import  cv2
import base64


def v2i(video_path):
    image_list=[]

    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    success = True  # 标志位，检查是否成功读取到视频帧
    frame_count=0
    while success:
        success, frame = cap.read()  # 读取下一帧
        if success:
            # 构造图像文件的保存路径
            # image_path = os.path.join("output", '{}.png'.format(frame_count))
            # cv2.imwrite(image_path, frame)  # 保存图像
            frame_count += 1
            image_list.append(frame)

            # print(str(frame_count)+"分割完毕")

    cap.release()  # 释放视频捕获对象
    return image_list


def find_purple_bounds(image):
    """
    寻找图片中紫色区域的最左上和最右下的像素点。
    """
    left, top, right, bottom = image.width, image.height, -1, -1
    found_purple = False
    purple_color=np.array([[100, 0, 100], [180, 60, 180]])
    for x in range(image.width):
        for y in range(image.height):
            rgb=image.getpixel((x, y))

            if (purple_color[0][0]<=rgb[0] and rgb[0]<=purple_color[1][0] and purple_color[0][1]<=rgb[1] and rgb[1]<=purple_color[1][1] and purple_color[0][2]<=rgb[2] and rgb[2]<=purple_color[1][2]):
                if not found_purple:
                    left, top = x, y
                    found_purple = True
                right, bottom = max(right, x), max(bottom, y)
    width, height = image.size
    top_left = (width, height)
    bottom_right = (0, 0)
    target_range = [(120, 0, 120), (140, 20, 140)]
    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            if all(target_range[0][i] <= pixel_value[i] <= target_range[1][i] for i in range(3)):
                top_left = (min(top_left[0], x), min(top_left[1], y))
                bottom_right = (max(bottom_right[0], x), max(bottom_right[1], y))

    if found_purple:
        return left, top, right, bottom

    else:
        return None


def split_image(image, rows=1, cols=2):
    """
    将图片分割成指定数量的块。
    """
    if image is None:
        return []

    height,width,_ = (image.shape)
    row_height = height // rows
    col_width = width // cols
    return [image[i*row_height:(i+1)*row_height,j*col_width: (j+1)*col_width,:] for i in range(rows) for j in range(cols)]


def decode_list(image_list, outfold):
    """
    遍历文件夹中的所有二维码图片，解码，可以将结果写入指定的输出文件。
    """
    decoded_list = []
    decoded_results = {}
    ih, iw, _ = image_list[0].shape
    # sizew=int(iw*0.03)
    # sizeh=int(ih*0.02)
    sizew = int(iw * 0)
    sizeh = int(ih * 0)
    bounds = (sizeh, sizew, ih - sizeh, iw - sizew)
    print(f"共识别{len(image_list)}帧，需解{2 * len(image_list)}张码")
    for i in range(len(image_list)):
        # image = Image.open(image_path)
        image = image_list[i]
        if bounds == None:
            bounds = find_purple_bounds(image)  # 获取紫色区域位置信息
            # print(bounds)

        if bounds:
            y0, x0, y1, x1 = bounds  # 解包位置信息
            purple_area = image[y0:y1, x0:x1, :]

            split_images = split_image(purple_area)
            decoded_list.extend(split_images)

    # 创建一个临时目录
    with TemporaryDirectory() as temp_dir:
        for i, img in enumerate(decoded_list):
            try:
                # 为每个图像创建唯一的临时文件路径
                temp_img_path = os.path.join(temp_dir, f'temp_{i}.png')
                # 保存numpy数组为图像文件
                cv2.imwrite(temp_img_path, img)
                # 解码临时文件
                begin=time.time()
                decoded_img = decode(temp_img_path)
                end=time.time()
                print(f"解第{i+1}张码耗时{end-begin:.4f}秒") # 目前大概 6 个/s
                index = int(decoded_img[:8])
                content = decoded_img[8:]
                if index not in decoded_results:
                    decoded_results[index] = content
            except:
                cv2.imwrite(outfold + "/test_{}.png".format(i), img)
                continue

    return decoded_results


def write_file(bytes_dict,output_folder,fd):
    result = bytes()
    for i in range(len(bytes_dict)):
        result += bytes_dict[i]
    filetype = result[:4].decode('utf-8').lstrip('0')
    padding = int(result[4:8].decode('utf-8'))
    content = result[8:len(result) - padding]

    file_path = os.path.join(output_folder, f"file.{filetype}")
    with open(file_path, "wb") as f:
        f.write(content)




