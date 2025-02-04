import cv2
import base64
# img1=cv2.imread("vediotran/encode/input/1.png")
# img2=cv2.imread("vediotran/decode/output/decode0.png")
# print(img1[0,0,0],img1[1,0,0])
# print(img2[0,0,0],img2[1,0,0])
# s
# print(img1)
with open("vediotran/encode/input/input.zip","rb") as img1:
    # im=base64.b64encode(img1.read())
    im=img1.read()
im=base64.b64encode(im).decode("utf-8")
print(im[:100])
print(im[-100:])
print(len(im))
im=base64.b64decode(im)
# print(im)
# # im=im.encode("utf-8")
print(type(im))
# # print(base64.b64decode(im))
with open("vediotran/encode/input/33.zip","wb") as img2:
    img2.write(im)
# print(img1.encode('utf-8'))
# print(im.decode('utf-8'))