import matplotlib.pyplot as plt
import h5py
import numpy as np

from utils.model_utils import *
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d  # 曲线平滑处理

COLORS = list(mcolors.TABLEAU_COLORS)
MARKERS = ["o", "v", "s", "*", "x", "p"]

plt.rcParams.update({'font.size': 14})
n_seeds = 3


def load_results(args, algorithm, mode, seed, time):
    if not algorithm.lower() == 'fedag':
        mode = 'all'
    alg = get_log_path(args, algorithm, mode, seed, time, args.gen_batch_size)
    hf = h5py.File("./{}/results_without_time/plot_iteration_results/{}.h5".format(args.result_path, alg), mode='r')
    # /plot_overall_results/
    # /plot_iteration_results/
    metrics = {}
    for key in METIRCS:
        if hf.get(key):
            metrics[key] = np.array(hf.get(key))
        else:
            print(f"{key} is none")
    return metrics


def get_label_name(name):
    name = name.split("-")[0]
    if "Distill" in name:
        if "-FL" in name:
            name = "FedDistill" + r'$^+$'
        else:
            name = 'FedDistill'
    elif "FedDF" in name:
        name = "FedFusion"
    elif "FedAvg" in name:
        name = "FedAvg"
    return name


def load_count(args):  # 加载保存客户端样本信息的文件（pt格式）
    dataset_ = args.dataset.split("-")  # 确定数据集信息(包含名称和alpha,ratio)
    alpha, ratio = dataset_[1], dataset_[2]  # 获取alpha，ratio值
    types = 'letters'  # EMnist数据集读取形式（此处只用于标记信息）
    if 'mnist' in args.dataset.lower():  # 根据数据集名称确定先前文件保存路径
        count_path = os.path.join('data', 'Mnist', f'u20c10-{alpha}-{ratio}')
    elif 'cifar10' in args.dataset.lower():
        count_path = os.path.join('data', 'CIFAR10', f'u20c10-{alpha}-{ratio}')
    elif 'emnist' in args.dataset.lower():
        count_path = os.path.join('data', 'EMnist', f'u20-{types}-{alpha}-{ratio}')
    else:
        raise ValueError("Dataset not recognized.")
    # 具体文件保存位置
    count_dir = os.path.join(count_path, 'count', 'count.pt')
    # 加载文件并返回其中内容，为以字典为元素的列表，长度为客户端总数量
    with open(count_dir, 'rb') as f:
        count = torch.load(f)
    return count


def plot_non_iid_sns(args):  # 用sns绘制non_iid可视化图像（散点图）（不建议使用，对比不明显）
    dataset_ = args.dataset.split('-')  # 读取数据集名称
    alpha = dataset_[1]  # alpha值
    subdir = dataset_[0]  # 保存路径文件夹名称
    plt.figure(1, figsize=(6, 5))  # 绘图数量及大小
    count = load_count(args)  # 根据输入参数加载图像
    client_count = [[] for _ in range(args.num_users)]  # 用于从原列表中各元素（字典）中加载各客户端类别数量列表，形式为以列表为元素的列表
    for i in range(args.num_users):
        client_count[i] = count[i]['train_idx_batch_length']  # 按所有用户数量依次加载
    class_count = np.array(client_count).reshape(args.num_users, -1)  # 将刚才列表形状规整为(客户端数量，类别数量)
    # weight=class_count/class_count.sum(axis=0,keepdims=True) 暂时未使用上，用于散点权重size
    print(class_count[:, 0])  # 测试语句
    for i in range(class_count.shape[1]):  # 在所有类别中循环
        x = np.arange(0, args.num_users) + 1  # 创建x数组，长度为客户端数量，值为各客户端的索引，用于绘图x轴范围
        y = np.array([i + 1] * args.num_users)  # 创建y数组，长度为客户端数量，数值为i+1,随循环变化。用于对应x中每个元素在散点图中的位置，使散点图整数点（对应客户端和类别）均有散点
        size = class_count[:, i]  # size用于设计散点大小
        index = []  # 用于存储去除不含该类i的客户端的索引
        for l, j in enumerate(class_count[:, i]):  # 对于所有第i类的样本数量
            if j == 0:  # 如果其中位置l的元素为0，即第l个客户端不包含该第i类样本
                index.append(l)  # 则添加该客户端索引至index列表
        x = np.delete(x, index)  # 删除x数组中指定位置的x
        y = np.delete(y, index)  # 对应删除数组中指定位置的y（i+1）
        if index != []:  # 如果index列表不为空
            size = np.delete(class_count[:, i], index)  # 删除size中指定索引位置的值（即权重）（不删会报错：数组长度不一致）
        ###
        # size权重归一化处理，散点区别不够明显
        size_avg = size.mean()
        size_std = size.std()
        size = (size - size_avg) / size_std
        print(x, y, size)
        ###
        # 每轮（类别）绘制散点图
        sns.scatterplot(x=x,
                        y=y,
                        size=10 * 4 ** size,  # 权重，散点大小
                        marker=MARKERS[0],  # 散点形状
                        color=COLORS[0],
                        legend=False,
                        )
    # 获取当前图像
    plt.gcf()
    # 显示表格线
    plt.grid()
    # x轴标记
    plt.xlabel("Client Index")
    # y轴标记
    plt.ylabel("Class Index")
    # 非iid数据分布
    plt.title("Non-IID Data Distribution")
    plt.show()
    # 图像保存路径
    save_path = f"figs/{subdir}/non-iid"
    # 图像保存路径不存在则创建
    if not os.path.exists(save_path):
        os.system('mkdir -p {}'.format(save_path))
    # 保存图像
    print("non_iid_distribution fig has been saved to：{}".format(os.path.join(save_path, alpha + '.pdf')))
    # dpi规定保存图像分辨率
    plt.savefig(os.path.join(save_path, alpha + '.pdf'), dpi=400)


def plot_non_iid_plt(args):  # 使用matplotlib创建non_iid可视化散点图（推荐使用，对比明显）
    dataset_ = args.dataset.split('-')  # 读取数据集名称
    alpha = dataset_[1]  # alpha值
    subdir = dataset_[0]  # 保存路径文件夹名称
    plt.figure(1, figsize=(7, 5))  # 绘图数量及大小
    count = load_count(args)  # 根据输入参数加载图像
    client_count = [[] for _ in range(args.num_users)]  # 用于从原列表中各元素（字典）中加载各客户端类别数量列表，形式为以列表为元素的列表
    for i in range(args.num_users):
        client_count[i] = count[i]['train_idx_batch_length']  # 按所有用户数量依次加载
    class_count = np.array(client_count).reshape(args.num_users, -1)  # 将刚才列表形状规整为(客户端数量，类别数量)
    # weight=class_count/class_count.sum(axis=0,keepdims=True) 暂时未使用上，用于散点权重size
    print(class_count[:, 0])  # 测试语句
    for i in range(class_count.shape[1]):  # 在所有类别中循环
        x = np.arange(0, args.num_users) + 1  # 创建x数组，长度为客户端数量，值为各客户端的索引，用于绘图x轴范围
        y = np.array([i + 1] * args.num_users)  # 创建y数组，长度为客户端数量，数值为i+1,随循环变化。用于对应x中每个元素在散点图中的位置，使散点图整数点（对应客户端和类别）均有散点
        size = class_count[:, i]  # size用于设计散点大小
        index = []  # 用于存储去除不含该类i的客户端的索引
        for l, j in enumerate(class_count[:, i]):  # 对于所有第i类的样本数量
            if j == 0:  # 如果其中位置l的元素为0，即第l个客户端不包含该第i类样本
                index.append(l)  # 则添加该客户端索引至index列表
        x = np.delete(x, index)  # 删除x数组中指定位置的x
        y = np.delete(y, index)  # 对应删除数组中指定位置的y（i+1）
        if index != []:  # 如果index列表不为空
            size = np.delete(class_count[:, i], index)  # 删除size中指定索引位置的值（即权重）（不删会报错：数组长度不一致）
        ###
        # size权重归一化处理，散点区别不够明显
        # size_avg=size.mean()
        # size_std=size.std()
        # size=(size-size_avg)/size_std
        print(x, y, size)
        ###
        plt.scatter(x=x,
                    y=y,
                    s=size,
                    marker=MARKERS[0],
                    color=COLORS[0]
                    )
    # 获取当前图像
    plt.gcf()
    # 显示表格线
    # plt.grid()
    # x轴标记
    plt.xlabel("Client Index")
    # y轴标记
    plt.ylabel("Class Index")
    # 规定x轴，y轴刻度(均为整数)
    plt.xticks(np.arange(1, args.num_users + 1))
    plt.yticks(np.arange(1, class_count.shape[1] + 1))
    # 非iid数据分布
    plt.title("Non-IID Data Distribution")
    # plt.show()
    # 图像保存路径
    save_path = f"figs/{subdir}/non-iid"
    if not os.path.exists(save_path):
        os.system('mkdir -p {}'.format(save_path))
    # 保存图像
    print("non_iid_distribution fig has been saved to：{}".format(os.path.join(save_path, alpha + '.pdf')))
    # dpi规定保存图像分辨率
    plt.savefig(os.path.join(save_path, alpha + '.pdf'), bbox_inches='tight', dpi=400)


def plot_results(args, algorithms):  # 绘图主函数
    time = '' # time需自行输入，以便获取正确实验结果路径
    n_seeds = args.times  # seed数量即为重复次数
    dataset_ = args.dataset.split('-')  # 按“-”分割数据集名称（字符串）
    sub_dir = dataset_[0] + "/" + dataset_[2]  # 即Mnist/ratio0.5 #！！！windows环境下路径改变为\\
    os.system("mkdir -p figs/{}".format(sub_dir))  # 即 figs/Mnist/ratio0.5
    plt.figure(1, figsize=(5, 5))  # 创造一个新图像，大小为（5，5）
    TOP_N = 5  # 最高精度的前5个
    max_acc = 0  # 最大精度
    # algorithms为列表
    for i, algorithm in enumerate(algorithms):  # 对于所有输入算法
        algo_name = get_label_name(algorithm)  # 获得不同算法对应的名称
        #######plot test accuracy##########
        # 按照seed值依次获取实验结果字典
        metrics = [load_results(args, algorithm, args.mode, seed, time) for seed in range(n_seeds)]
        # 将不同seed字典中的键'glob_acc'的对应值拼接起来，并将其转换为numpy数组
        all_curves = np.concatenate([metrics[seed]['glob_acc'] for seed in range(n_seeds)])
        # 将不同seed中全局精度列表中值最高的前TOP_N个值保存在top_accs数组中，np.sort默认按从小到大顺序排序
        top_accs = np.concatenate([np.sort(metrics[seed]['glob_acc'])[-TOP_N:] for seed in range(n_seeds)])
        acc_avg = np.mean(top_accs)  # 取精度平均值
        acc_std = np.std(top_accs)  # 取精度标准差
        print(len(all_curves))
        # 构建需要打印的字符串，其中<10s代表左对齐的字符串，宽度为10，不足部分为空，超出部分截断
        info = 'Algorithm: {:<10s}, Accuracy={:.4f} , deviation={:.4f}'.format(algorithm, acc_avg, acc_std)
        # 打印信息：算法，精度，标准差
        print(info)
        # 每个精度数组/3的长度，即一次训练的总轮数
        length = len(all_curves) // n_seeds

        #window_size = 3
        # 使用滑动窗口平滑曲线，即对精度数组进行卷积，卷积核为window_size个1，卷积模式为valid，即不填充，只保留卷积结果
        # 卷积结果长度为精度数组长度减去卷积核长度+1
        #all_curves_smoothed = np.convolve(all_curves, np.ones(window_size) / window_size, mode='same')
        #print(len(all_curves_smoothed))
        #all_curves_smoothed = gaussian_filter1d(all_curves, sigma=1)
        # 使用seaborn绘制第i个算法的折线图

        sampled_indices = range(0, len(all_curves), 5)
        sampled_data = all_curves[sampled_indices]
        x = np.array(list(range(length)) * n_seeds) + 1

        sns.lineplot(
            x=x[sampled_indices],
            # x轴内容为从1到length+1数组的重复n_seeds次，如[1,2,...,200,1,2,...,200,1,2,...,200]
            y=sampled_data.astype(float),  # y轴内容为精度数组
            legend='brief',  # 图例，‘brief'表示使用简洁模式，图例中不显示每个序列的具体名称和坐标系标签只显示类别名
            color=COLORS[i],  # 颜色按序从0-9，即从蓝色到深红色
            label=algo_name,  # 图形下方的标签为算法名字
            errorbar=None,
            #ci="sd",  # 表示置信区间(confidence interval)的宽度 int or "sd" or None，sd表示标准差（standard deviation)

        )  # sns.lineplot绘制的折线图带有阴影部分，表示置信区间。
        # 打印的折线图x轴范围即为1到length+1，实际y轴数值为n_seeds个不同实验的平均值

    plt.gcf()  # gcf=get the current figure即获取当前图表
    plt.grid()  # 显示网格线，默认为1=True=显示
    plt.title(dataset_[0] + ' Test Accuracy')  # 标题为数据集名称+'Test Accuracy'
    plt.xlabel('Epoch')  # x轴标签
    max_acc = np.max([max_acc, np.max(all_curves)]) + 4e-2  # 获取最大精度并+4e-2，此句是否应该放在循环内

    if args.min_acc < 0:  # 如果输入的最小精度小于0
        alpha = 0.4
        # 则y轴的最小精度为最大精度乘以alpha加上最小精度乘以(1-alpha)
        min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1 - alpha)
    else:
        min_acc = args.min_acc
    # y轴的取值范围为min_acc到max_acc
    plt.ylim(min_acc, max_acc)
    # 图表保存路径：如：figs/Mnist/ratio0.5/Mnist-alpha0.1.png#此处改动为dataset_[1]源码为[2]
    fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + args.mode + '.pdf')
    # 保存图表。
    # bbox_inches表示自动裁剪边界框，将图形进行一定的缩小和放大，以便去除不必要的空白区域，设定为’tight'以压缩留白区域，同时允许使用pad_inches参数
    # pad_inches表示边距，定义图像在边界框内的间距，正值在四周增加额外空间，负值则从四周减去空间，默认值为0.1
    # format表示保存格式，dpi表示分辨率，表示像素/英寸，默认为100
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=400)
    # 打印输出路径
    print("file saved to {}".format(fig_save_path))

def plot_server_results(args, algorithms):
    time = ''  # time需自行输入，以便获取正确实验结果路径
    n_seeds = args.times  # seed数量即为重复次数
    dataset_ = args.dataset.split('-')  # 按“-”分割数据集名称（字符串）
    sub_dir = dataset_[0] + "/" + dataset_[2]  # 即Mnist/ratio0.5 #！！！windows环境下路径改变为\\
    os.system("mkdir -p figs/{}".format(sub_dir))  # 即 figs/Mnist/ratio0.5
    plt.figure(2, figsize=(5, 5))  # 创造一个新图像，大小为（5，5）
    TOP_N = 5  # 最高精度的前5个
    max_acc = 0  # 最大精度
    # algorithms为列表
    for i, algorithm in enumerate(algorithms):  # 对于所有输入算法
        algo_name = get_label_name(algorithm)  # 获得不同算法对应的名称
        #######plot test accuracy##########
        # 按照seed值依次获取实验结果字典
        metrics = [load_results(args, algorithm, args.mode, seed, time) for seed in range(n_seeds)]
        # 将不同seed字典中的键'glob_acc'的对应值拼接起来，并将其转换为numpy数组
        all_curves = np.concatenate([metrics[seed]['server_acc'] for seed in range(n_seeds)])
        # 将不同seed中全局精度列表中值最高的前TOP_N个值保存在top_accs数组中，np.sort默认按从小到大顺序排序
        top_accs = np.concatenate([np.sort(metrics[seed]['server_acc'])[-TOP_N:] for seed in range(n_seeds)])
        acc_avg = np.mean(top_accs)  # 取精度平均值
        acc_std = np.std(top_accs)  # 取精度标准差
        print(len(all_curves))
        # 构建需要打印的字符串，其中<10s代表左对齐的字符串，宽度为10，不足部分为空，超出部分截断
        info = 'Algorithm: {:<10s}, Accuracy={:.4f} %, deviation={:.4f}'.format(algorithm, acc_avg, acc_std)
        # 打印信息：算法，精度，标准差
        print(info)
        # 每个精度数组/3的长度，即一次训练的总轮数
        length = len(all_curves) // n_seeds

        sampled_indices = range(0, len(all_curves), 5)
        sampled_data = all_curves[sampled_indices]
        x = np.array(list(range(length)) * n_seeds) + 1

        # 使用seaborn绘制第i个算法的折线图
        sns.lineplot(
            #x=np.array(list(range(length)) * n_seeds) + 1,
            x=x[sampled_indices],
            # x轴内容为从1到length+1数组的重复n_seeds次，如[1,2,...,200,1,2,...,200,1,2,...,200]
            #y=all_curves.astype(float),  # y轴内容为精度数组
            y=sampled_data.astype(float),
            # legend='brief',  # 图例，‘brief'表示使用简洁模式，图例中不显示每个序列的具体名称和坐标系标签只显示类别名
            legend=False, # 手动设置图例位置
            color=COLORS[i],  # 颜色按序从0-9，即从蓝色到深红色
            label=algo_name,  # 图形下方的标签为算法名字
            errorbar=None,  # 表示置信区间(confidence interval)的宽度 int or "sd" or None，sd表示标准差（standard deviation)
        )  # sns.lineplot绘制的折线图带有阴影部分，表示置信区间。
        # 打印的折线图x轴范围即为1到length+1，实际y轴数值为n_seeds个不同实验的平均值
        plt.legend(bbox_to_anchor=(0.5, 0.6)) #设置图例左下角坐标位置
    plt.gcf()  # gcf=get the current figure即获取当前图表
    plt.grid()  # 显示网格线，默认为1=True=显示
    plt.title(dataset_[0] + ' Server Test Accuracy')  # 标题为数据集名称+'Test Accuracy'
    plt.xlabel('Epoch')  # x轴标签
    max_acc = np.max([max_acc, np.max(all_curves)]) + 4e-2  # 获取最大精度并+4e-2，此句是否应该放在循环内

    if args.min_acc < 0:  # 如果输入的最小精度小于0
        alpha = 0.4
        # 则y轴的最小精度为最大精度乘以alpha加上最小精度乘以(1-alpha)
        min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1 - alpha)
    else:
        min_acc = args.min_acc
    # y轴的取值范围为min_acc到max_acc
    plt.ylim(min_acc, max_acc)
    # 图表保存路径：如：figs/Mnist/ratio0.5/Mnist-alpha0.1.png#此处改动为dataset_[1]源码为[2]
    fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + args.mode + 'server' + '.pdf')
    # 保存图表。
    # bbox_inches表示自动裁剪边界框，将图形进行一定的缩小和放大，以便去除不必要的空白区域，设定为’tight'以压缩留白区域，同时允许使用pad_inches参数
    # pad_inches表示边距，定义图像在边界框内的间距，正值在四周增加额外空间，负值则从四周减去空间，默认值为0.1
    # format表示保存格式，dpi表示分辨率，表示像素/英寸，默认为100
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=400)
    # 打印输出路径
    print("file saved to {}".format(fig_save_path))
