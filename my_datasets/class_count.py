import json
from collections import defaultdict

with open('tryy_with_normal.jsonl', 'r', encoding='utf-8') as f:
    # 统计每个类别的数量
    class_count = defaultdict(int)
    for line in f:
        item = json.loads(line)
        try:
            #class_count[item['modality']] += 1
            if item['modality'] == 'color':
                class_count[item['class']] += 1
        except:
            print("Error parsing line:", line)
    print(len(class_count))

# 计算个类别比例
total = sum(class_count.values())
class_ratio = {cls: count / total for cls, count in class_count.items()}
print(class_ratio)
