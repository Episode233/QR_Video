# -*- coding: utf-8 -*-
import time, multiprocessing, tempfile, os, cv2
from cimbar.cimbar import decode
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
import numpy as np


def v2i(video_path):
    image_list = []

    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    success = True  # 标志位，检查是否成功读取到视频帧
    frame_count = 0
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
    purple_color = np.array([[100, 0, 100], [180, 60, 180]])
    for x in range(image.width):
        for y in range(image.height):
            rgb = image.getpixel((x, y))

            if (purple_color[0][0] <= rgb[0] and rgb[0] <= purple_color[1][0] and purple_color[0][1] <= rgb[1] and rgb[
                1] <= purple_color[1][1] and purple_color[0][2] <= rgb[2] and rgb[2] <= purple_color[1][2]):
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

    height, width, _ = (image.shape)
    row_height = height // rows
    col_width = width // cols
    return [image[i * row_height:(i + 1) * row_height, j * col_width: (j + 1) * col_width, :] for i in range(rows) for j
            in range(cols)]


def process_single_image(args):
    """Process a single image in a separate process with optimized memory handling"""
    i, img, temp_dir = args
    temp_img_path = None
    try:
        # Create a unique temp file - only if decode function requires a file path
        fd, temp_img_path = tempfile.mkstemp(suffix='.png', dir=temp_dir)
        os.close(fd)  # Close the file descriptor

        # Save the image with optimized compression parameters for speed
        # Use PNG compression level 1 (fastest) instead of default
        cv2.imwrite(temp_img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        begin = time.time()
        # Perform decoding
        decoded_img = decode(temp_img_path)
        end = time.time()

        print(f"解第{i + 1}张码耗时{end - begin:.4f}秒")

        # Clean up the temp file immediately after use
        if os.path.exists(temp_img_path):
            os.unlink(temp_img_path)

        if decoded_img:
            # Extract index and content
            index = int(decoded_img[:8])
            content = decoded_img[8:]
            return (index, content)

        return None
    except Exception as e:
        print(f"处理图像 {i + 1} 时出错: {str(e)}")
        # Ensure cleanup even in case of error
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.unlink(temp_img_path)
            except:
                pass
        return None


def decode_list(image_list, outfold):
    """
    使用多进程并行解码图像列表，优化进程效率和内存使用
    """
    decoded_results = {}

    # 如果列表为空，直接返回
    if not image_list:
        return decoded_results

    ih, iw, _ = image_list[0].shape
    sizew = int(iw * 0)
    sizeh = int(ih * 0)
    bounds = (sizeh, sizew, ih - sizeh, iw - sizew)

    # 计算图像总数
    total_images = len(image_list) * 2
    print(f"共识别{len(image_list)}帧，需解{total_images}张码")

    # 处理图像分割 - 优化分割逻辑
    decoded_list = []

    # 预分配内存可以提高性能
    decoded_list = [None] * (len(image_list) * 2)
    index = 0

    for i in range(len(image_list)):
        image = image_list[i]

        # 使用切片直接分割图像，避免找边界的开销
        if bounds:
            y0, x0, y1, x1 = bounds
            purple_area = image[y0:y1, x0:x1, :]

            # 直接分割成两半
            height, width, _ = purple_area.shape
            mid_width = width // 2

            # 直接添加到列表中
            decoded_list[index] = purple_area[:, :mid_width, :]
            index += 1
            decoded_list[index] = purple_area[:, mid_width:, :]
            index += 1

    # 移除未填充的元素
    decoded_list = decoded_list[:index]

    # 自动确定最佳进程数
    cpu_count = multiprocessing.cpu_count()
    # 基于CPU密集型任务特性，使用更多内核
    available_cores = max(1, cpu_count - 1)
    # 计算合适的进程数
    num_processes = min(available_cores, len(decoded_list))

    print(f"系统检测到 {cpu_count} 个CPU核心，将使用 {num_processes} 个进程进行并行解码")

    # 创建一个临时目录
    with TemporaryDirectory() as temp_dir:
        # 优化：使用更大的批量大小
        batch_size = max(1, len(decoded_list) // (num_processes * 2))

        # 准备任务参数 - 批量处理
        task_args = []
        for i, img in enumerate(decoded_list):
            task_args.append((i, img, temp_dir))

        # 使用进程池执行解码任务
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 使用 chunksize 参数优化任务分配
            results = list(executor.map(process_single_image, task_args, chunksize=batch_size))

            # 处理结果
            for result in results:
                if result:
                    index, content = result
                    if index not in decoded_results:
                        decoded_results[index] = content

    # TemporaryDirectory 上下文管理器会自动清理临时目录
    return decoded_results

def write_file(bytes_dict, output_folder, fd):
    result = bytes()
    for i in range(len(bytes_dict)):
        result += bytes_dict[i]
    filetype = result[:4].decode('utf-8').lstrip('0')
    padding = int(result[4:8].decode('utf-8'))
    content = result[8:len(result) - padding]

    file_path = os.path.join(output_folder, f"file.{filetype}")
    with open(file_path, "wb") as f:
        f.write(content)
