from PIL import Image
from pyzbar.pyzbar import decode
import cv2
import numpy as np
def find_purple_bounds(image):
    """
    寻找图片中紫色区域的最左上和最右下的像素点。
    """
    height,width,_=image.shape
    left, top, right, bottom = width/2, height/2, width/2, height/2
    found_purple = False
    purple_color=np.array([[100, 0, 100], [180, 60, 180]])
    for x in range(width):
        for y in range(height):
            rgb=image[y, x]
            if (230<=rgb[0] and 230<=rgb[1] and 230<=rgb[2] ):
                    if(left>x):
                        left=x
                    if(top>y):
                        top=y
                    if(right<x):
                        right=x
                    if(bottom<y):
                        bottom=y
                    
     
    return left, top, right, bottom
    

img=cv2.imread("vediotran/decode/output/test1001_3.png")
print(find_purple_bounds(img))
# ih,iw,_=img.shape
# masize=8#一张图8张二维码
# sizew=int(iw*0.01)
# sizeh=int(ih*0.02)
# bounds=(sizeh,sizew,int(ih-ih*0.1),iw-sizew)
# x0, y0, x1, y1 = find_purple_bounds(img)  # 解包位置信息
# purple_area = img[y0:y1,x0:x1,:]
decoded = decode(img)
print(img)
print(decoded)
# cv2.imwrite("vediotran/decode/output/tt.png",purple_area)