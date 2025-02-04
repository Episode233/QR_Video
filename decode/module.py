# -*- coding: utf-8 -*-
import cv2
#pip install Pillow pyzbar
import os
from PIL import Image
from pyzbar.pyzbar import decode
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
            
            print(str(frame_count)+"分割完毕")

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

def split_image(image, rows=2, cols=4):
    """
    将图片分割成指定数量的块。
    """
    if image is None:
        return []

    height,width,_ = (image.shape)
    row_height = height // rows
    col_width = width // cols
    return [image[i*row_height:(i+1)*row_height,j*col_width: (j+1)*col_width,:] for i in range(rows) for j in range(cols)]
def decode_list(image_list,outfold):
    """
    遍历文件夹中的所有二维码图片，解码，并将结果写入指定的输出文件。
    """
    decoded_results = {}
    ih,iw,_=image_list[0].shape
    masize=8#一张图8张二维码
    # sizew=int(iw*0.03)
    # sizeh=int(ih*0.02)
    sizew=int(iw*0)
    sizeh=int(ih*0)
    bounds=(sizeh,sizew,ih-sizeh,iw-sizew)
    # bounds=None
    iter=-1#每1000张一次迭代
    flag=1#表示是否修改迭代次数
    for i in range(len(image_list)):
        image_id=None
        code_8={}
        # image = Image.open(image_path)
        image=image_list[i]
        if bounds==None:
                bounds = find_purple_bounds(image)  # 获取紫色区域位置信息
                # print(bounds)
                
        if bounds:
                y0, x0, y1, x1 = bounds  # 解包位置信息
                purple_area = image[y0:y1,x0:x1,:]
             
                split_images = split_image(purple_area)
                if i==0:
                    for img in range(len(split_images)):
                        cv2.imwrite(outfold+"/test_{}.png".format(img),split_images[img] )
                        # split_images[img].save(outfold+"/test_{}.png".format(img))
                for idx, img_part in enumerate(split_images):
                     
                     decoded = decode(img_part)
                     
                     if(len(decoded)==0 or len(decoded[0].data.decode('utf-8'))<400):
                         continue
                     else:
                         decodetxt=decoded[0].data.decode('utf-8')[3:]
                         
                         image_id=int(decoded[0].data.decode('utf-8')[:3])
                         code_8[idx]=decodetxt

                         
                        #  print(decoded[0].data.decode('utf-8'))
                        #  print(len(decoded_results),int(image_id),idx)
                         if(image_id!=None):
                            if(int(image_id )==0):
                                iter=-1
                                flag=1
                            elif int(image_id )<10 and flag==1:
                                iter+=1
                                flag=0
                                image_id=999*iter+image_id
                            elif int(image_id) >990 and flag==0:#末尾修正flag，保证只在头部进行iter+1
                                flag=1
                                image_id=999*iter+image_id
                            else:
                                image_id=999*iter+image_id
                         if(int(image_id) in decoded_results and len(decoded_results[int(image_id)])==8):
                             break
                if(image_id!=None):
                    
                    if(int(image_id )==25):
                        for img in range(len(split_images)):
                            cv2.imwrite(outfold+"/test1001_{}.png".format(img),split_images[img] )
                    if(int(image_id) not in decoded_results):
                        
                        decoded_results[int(image_id)]=code_8
                    else:
                        if(len(decoded_results[int(image_id)])<8):
                            print(int(image_id))
                            for index in range(8):
                                if (index not in decoded_results[int(image_id)] and index in code_8):
                                    decoded_results[int(image_id)][index]=code_8[index]
                    if(len(decoded_results[int(image_id)])==8):
                        print(f"识别成功{int(image_id)}")
    sunum=0
    index=0             
    unsulist=[]     
    for image in decoded_results:
        if(len(decoded_results[image])==8):
            print(f"第{index+1}张识别成功,共{len(decoded_results)}张图像,共识别成功{sunum+1}张")
            sunum+=1
        else:
            unsulist.append(index+1)
        index+=1
    if(sunum==index or sunum==index-1):
        print("完全识别成功")
    else:
        for i in unsulist:
            print(f"第{i}张识别失败")
    return decoded_results
    
  

def write_file(textdict,outputflod,fd):    
    str=""
    for i in range(len(textdict)):
        for image in range(len(textdict[i])):
            str+=textdict[i][image]
    filetype=str[:4]
    while(True):
        if(filetype[0]=="0"):
            filetype=filetype[1:]
        else:
            break
    while(True):
        if(str[-1]=="="):
            str=str[:-1]
        else:
            break
    if(len(str)%4==3):
        str+="="
    elif(len(str)%4==2):
        str+="=="
    elif(len(str)%4==1):
        str+="==="
    

    outfile=outputflod+"/decode{}.".format(int(fd))+filetype
    str=str[4:]
    with open(outfile,"wb") as img:
        img.write(base64.b64decode(str))




    

