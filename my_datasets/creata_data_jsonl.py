data_dir = '/home/ubuntu/qujunlong/data/bacteria'
import os
from collections import Counter

# 获取所有子文件夹名称,即类别名称
categories = os.listdir(data_dir)

# 统计每个类别下的样本数量
category_counts = Counter()
for category in categories:
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        # 统计该类别文件夹下的文件数量
        files = os.listdir(category_path)
        category_counts[category] = len(files)

# 打印统计结果
print("\n类别统计结果:")
for category, count in category_counts.items():
    print(f"{category}: {count}个样本")
print(f"\n总共{len(category_counts)}个类别, {sum(category_counts.values())}个样本")


import json

# 创建jsonl文件
output_file = 'bacteria_dataset.jsonl'

with open(output_file, 'w', encoding='utf-8') as f:
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            files = os.listdir(category_path)
            for file in files:
                # 构建相对路径
                relative_path = os.path.join(category, file)
                # 创建数据字典
                data = {
                    'image_path': relative_path,
                    'category': category
                }
                # 写入jsonl文件
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"\nJsonl文件已创建: {output_file}")
