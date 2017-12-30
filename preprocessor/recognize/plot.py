import argparse
import numpy as np
from matplotlib import pyplot as plt


def plot(penalty_dict, save_dir):
    """Plot the penalty class vs. money/weight/BAC which occur in the text.

    Args:
        penalty_dict: <dict> Example:
            penalty_dict = {'1': [10, 13, 21, ...],
                            '2': [233, 420, 300, ...],
                            ...
                            '8': [5e6, 6e6, 7e7, ...]}
    """
    plt.suptitle('Penalty Class w.r.t. Money')
    for i in range(8):
        plt.subplot('81' + str(i+1))
        x = np.array(penalty_dict[str(i+1)])
        y = np.ones(x.size)
        plt.xscale('log')
        plt.ylabel('{}'.format(i+1))
        plt.yticks([])  # 关闭y轴坐标刻度。
        plt.xlim(1, 1e7)
        plt.stem(x, y, linefmt='r', markerfmt=' ', basefmt=' ')
    plt.savefig(save_dir + '/test.png')


if __name__ == '__main__':
    penalty_dict = {'1': [10, 233, 300],
                    '2': [560, 780, 3400],
                    '3': [5000, 6000, 7000],
                    '4': [2e4, 1e4, 4e4],
                    '5': [7e4, 8e4, 9e4],
                    '6': [2e5, 3e5, 4e5],
                    '7': [5e5, 6e5, 7e5],
                    '8': [5e6, 6e6, 7e7]}

    plot(penalty_dict, save_dir='.')
