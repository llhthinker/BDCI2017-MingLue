import re
import io
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt

from arsenal.regex import recognize
from arsenal.regex import convert

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--save-dir', type=str)
    args = parser.parse_args()

    penalty_dict = {}
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        penalty_dict[i] = []

    recognized_info = io.StringIO()

    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            id, text, penalty = line.split('\t')[0:3]
            text = convert.full2half(text)
            res = recognize.recognize_money(text)
            recognized_info.write(id + ' ' + json.dumps(res, ensure_ascii=False) + '\n')
            for r in res:
                penalty_dict[penalty].append(r['std_value'])

    with open(args.save_dir + '/money_info.txt', 'w') as f:
        f.write(recognized_info.getvalue().strip())

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
    plt.savefig(args.save_dir + '/money.png')
