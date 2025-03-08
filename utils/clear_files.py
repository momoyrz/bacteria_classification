import os
from glob import glob

file_list = glob('../train_output/*/*/log.txt')

# 如果file_list中文件小于200KB,获取上级目录
file_list = [file for file in file_list if os.path.getsize(file) < 200 * 1024]
file_list = [os.path.dirname(file) for file in file_list]
file_list = list(set(file_list))

# 删除file_list中的文件夹，以及../train_output/checkpoint/同名文件夹
for file in file_list:
    os.system(f'rm -rf {file}')
    os.system(f'rm -rf ../train_output/checkpoint/{os.path.basename(file)}')
