# from PIL import Image
# import PIL.ImageOps    
# import numpy as np

import cv2


def binarize_image_opencv(image_path, threshold, invert=True):
    # 打开图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式打开图像
    
    # 选择二值化标志
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    
    # 二值化处理
    ret, binary_img = cv2.threshold(img, threshold, 255, threshold_type)
    
    return binary_img

# 输入图片路径、阈值和颜色反转标志
input_image_path = "annotations_image.png"  # 替换为你的灰度图像路径
threshold = 128  # 你可以根据实际情况调整阈值
invert = True  # 设置为True进行颜色反转，设置为False则不反转

# 进行二值化处理
binarized_image = binarize_image_opencv(input_image_path, threshold, invert)

# 保存二值化后的图片
output_image_path = "binarized_image.png"
cv2.imwrite(output_image_path, binarized_image)




# from PIL import Image
# import torchvision.transforms as transforms
# import torch

# # 定义图像变换，包括归一化
# img_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(0.5, 0.5)
# ])

# # 加载灰度图像并进行处理
# gray_image_path = "annotations_image.png"  # 替换为你的灰度图像路径
# # gray_image_path = "./data/massroof0801/tr-bmc-1000/images/bmc_0.png"
# gray_image = Image.open(gray_image_path).convert("L")  # 确保是灰度图像
# normalized_gray_image = img_transform(gray_image)

# print("Normalized Gray Image Tensor:")
# print(normalized_gray_image)



# image = Image.open('annotations_image.png')
# image = image.convert('L')
# inverted_image = PIL.ImageOps.invert(image)
# inverted_image.save('inverted_image.png')