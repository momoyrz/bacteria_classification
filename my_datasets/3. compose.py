import os

import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch

csv_path = '/home/ubuntu/qujunlong/txyy/my_classification/my_datasets/处理zeiss-c500副本/output.csv'
df = pd.read_csv(csv_path)

# 读取图片列
image_files = df['name'].tolist()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 遍历每一张图片
for image_file in image_files:
    try:
        # 打开图像
        image = Image.open(image_file)
        # 应用转换
        image_tensor = transform(image)
        # 获取文件名和扩展名
        file_name, _ = os.path.splitext(image_file)
        # 保存tensor张量
        tensor_path = file_name + '.pt'
        torch.save(image_tensor, tensor_path)
        print(f"Saved tensor for {image_file} at {tensor_path}")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")