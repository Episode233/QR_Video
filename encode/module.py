import numpy as np, qrcode, cv2
from PIL import Image
from cimbar.cimbar import encode

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


chunk_size = 30 * 30
outPath = 'outPut'
qrMaxSizeDic = {
    qrcode.constants.ERROR_CORRECT_L: '2950',
    qrcode.constants.ERROR_CORRECT_M: '2330',
    qrcode.constants.ERROR_CORRECT_Q: '1660',
    qrcode.constants.ERROR_CORRECT_H: '1270'

}


def create_qrCode(filePath):
    imagelist = encode(filePath)
    return imagelist


def pingjie(imagelist):
    # 大图片尺寸
    big_image_size = (3840, 2160)
    # 每块大小
    block_width = big_image_size[0] // 2
    block_height = big_image_size[1]
    block_size = (block_width, block_height)
    # 小二维码尺寸
    small_qr_size = (1800, 1800)
    image_number = 2  # 一张图填充个数
    # 创建紫色填充图像
    purple_color = (128, 0, 128)
    # 遍历文件夹内的图片
    num = len(imagelist)
    imagelist_2 = []

    # 使用ThreadPoolExecutor而不是多进程
    from concurrent.futures import ThreadPoolExecutor
    import gc

    # 定义处理单个图像的函数
    def process_image(i):
        # 创建大图片
        big_image = Image.new('RGB', big_image_size, purple_color)

        for j in range(2):
            if (i * 2 + j + 1) > num:
                break
            # 打开小二维码
            small_qr = imagelist[i * 2 + j]
            # 缩放小二维码
            small_qr = small_qr.resize(small_qr_size)

            # 计算放置位置
            left = j * block_width
            top = 0

            # 计算二维码在紫色底片中的居中位置
            x_offset = (block_width - small_qr_size[0]) // 2
            y_offset = (block_height - small_qr_size[1]) // 2

            # 将二维码居中放置在紫色底片中
            big_image.paste(small_qr, (left + x_offset, top + y_offset))

        return big_image

    # 获取CPU核心数
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # 限制使用的核心数，预留一个核心给系统
    cpu_count = max(1, cpu_count - 1)

    # 使用线程池处理图像
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        # 处理每一组图像
        futures = [executor.submit(process_image, i) for i in range(0, num // 2 + 1)]

        # 收集结果
        for future in futures:
            result = future.result()
            if result:
                imagelist_2.append(result)

    # 强制进行垃圾回收
    gc.collect()

    return imagelist_2


def img2vedio(image_list, video_path):
    # 图像所在文件夹路径
    # 输出视频的路径
    video_path = video_path + '/output_video.avi'
    width, height = image_list[0].size

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 使用DIVX编码器
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    new_imagelist = []
    for image in image_list:
        new_imagelist.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    for i in range(3):
        video.write(new_imagelist[0])
    for image in new_imagelist[:]:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()