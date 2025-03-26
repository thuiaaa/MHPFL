import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def collect_and_plot(results_dir='results', output_name='comparison'):
    # 自动识别所有算法
    algorithms = ['FedMoEKD', 'pFedMoE', 'FedGen', 'FedDistill', 'FedProto']

    # 动态生成对比配置
    data_dict = {}
    global_min_length = float('inf')  # 全局最小长度

    # 第一次遍历：收集所有数据并确定全局最小长度
    for algo in sorted(algorithms):
        pattern = f"Cifar10_{algo}_*_*_*_acc.npy"
        file_paths = glob.glob(os.path.join(results_dir, pattern))

        if file_paths:
            acc_list = [np.load(f) for f in file_paths]
            min_len = min(len(arr) for arr in acc_list)
            global_min_length = min(global_min_length, min_len)
            data_dict[algo] = acc_list

    # 第二次遍历：统一截断并计算统计量
    for algo, acc_list in data_dict.items():
        # 统一截断到全局最小长度
        truncated = [arr[:global_min_length] for arr in acc_list]

        # 添加维度检查
        if len({arr.shape for arr in truncated}) > 1:
            raise ValueError(f"算法 {algo} 的数组形状不一致: {[arr.shape for arr in truncated]}")

        data_dict[algo] = {
            'mean': np.mean(truncated, axis=0),
            'std': np.std(truncated, axis=0)
        }

    # 绘图部分改用预处理数据
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

    for (algo, stats), color in zip(data_dict.items(), colors):
        plt.plot(stats['mean'], label=algo, color=color, linewidth=2)
        plt.fill_between(range(global_min_length),
                        stats['mean']-stats['std'],
                        stats['mean']+stats['std'],
                        color=color, alpha=0.1)

    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Algorithm Comparison ({len(data_dict)} Methods)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    collect_and_plot(results_dir="/home/MHPFL/system/results/")