import argparse
import time
import json

def merge_result(task1_result_path, task2_result_path):
    task1_res = open(task1_result_path)
    res1 = []
    for line in task1_res:
        res1.append(eval(line))


    task2_res = open(task2_result_path)
    res2 = []
    for line in task2_res:
        res2.append(eval(line))

    for i in range(len(res1)):
        res1[i]["laws"] = res2[i]["laws"]


    time_stamp = str(int(time.time()))
    outf = open("./results/merge_result"+"."+time_stamp, 'a')
    for line in res1:
        json.dump(line, outf)
        outf.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result1", type=str)
    parser.add_argument("--result2", type=str)
    args = parser.parse_args()

    merge_result(args.result1, args.result2)

