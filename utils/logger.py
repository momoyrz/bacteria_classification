import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # 日志格式, 日期, 名称, 文件名, 行号, 等级, 消息
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process

    console_handler = logging.StreamHandler(sys.stdout)     # 创建控制台处理器
    console_handler.setLevel(logging.DEBUG)                 # 设置控制台处理器的日志等级
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))        # 设置控制台处理器的日志格式
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log.txt'), mode='a')    # 创建文件处理器
    file_handler.setLevel(logging.DEBUG)                                                     # 设置文件处理器的日志等级
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))     # 设置文件处理器的日志格式
    logger.addHandler(file_handler)                                             # 将文件处理器添加到logger中

    return logger

