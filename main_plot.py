import h5py
import matplotlib.pyplot as plt
import numpy as np
import importlib
import random
import os
import argparse
from utils.plot_utils import *
import torch

torch.manual_seed(0)
# 设置随机种子

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集名称
    parser.add_argument("--dataset", type=str, default="Mnist-alpha0.1-ratio0.5")
    # 算法名称
    parser.add_argument("--algorithms", type=str, default="FedAvg,Fedgen", help="algorithm names separate by comma")
    # 实验结果保存路径
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--mode",type=str, default='all')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate.")
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_glob_iters", type=int, default=200)
    # 设定y轴最小精度
    parser.add_argument("--min_acc", type=float, default=-1.0)
    # 激活客户端数量
    parser.add_argument("--num_users", type=int, default=5, help="number of active users per epoch.")
    # 客户端batch size
    parser.add_argument("--batch_size", type=int, default=32)
    # 生成器batch_size
    parser.add_argument("--gen_batch_size", type=int, default=32)
    # 绘图图例大小为1,plot_utils未设置
    parser.add_argument("--plot_legend", type=int, default=1,
                        help="plot legend if set to 1, omitted otherwise(否则省略)")
    # 重复训练次数
    parser.add_argument("--times", type=int, default=3, help="number of random seeds, starting from 1")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    args = parser.parse_args()

    # a.strip()用于去除字符串首尾空格字符，可填充以去除指定字符
    algorithms = [a.strip() for a in args.algorithms.split(',')]
    # title貌似没有被使用
    title = 'epoch{}'.format(args.local_epochs)
    # plot_utils.py中的plot_results()函数
    # 按输入参数及经处理的算法名称绘图
    plot_results(args, algorithms)
    plot_server_results(args, algorithms)
    #plot_non_iid_plt(args)
