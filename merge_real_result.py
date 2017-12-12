import argparse
import time
import json

def merge_result(task_result_path, real_result_path):
    task_res = open(task_result_path)
    res = []
    for line in task_res:
        res.append(eval(line))
    
    with open(real_result_path, 'r') as real_f:
        real_result = json.load(real_f)
    # print(real_result.keys())

    total_count, cor_count = 0, 0
    cor_set_score = 0
    for i in range(len(res)):
        if res[i]["id"] in real_result:
            # print(res[i]["id"])
            total_count += 1 
            if res[i]["penalty"] == real_result[res[i]["id"]][0]:
                cor_count += 1
            cor_set_score += len(set(res[i]["laws"]) & set(real_result[res[i]["id"]][1])) / len(set(res[i]["laws"]) | set(real_result[res[i]["id"]][1]))
            res[i]["penalty"] = real_result[res[i]["id"]][0]
            res[i]["laws"] = real_result[res[i]["id"]][1]
    print("total count:", total_count)
    print("correct count:", cor_count)
    print("real acc:", cor_count / total_count)
    print("set score:", cor_set_score / total_count)
    time_stamp = str(int(time.time()))
    """
    outf = open("./results/merge_real_result"+"."+time_stamp, 'a')
    for line in res:
        # json.dump(line, outf)
        new_line = "{\"id\": \"" + line["id"] + "\", \"penalty\": "+ str(line["penalty"]) + ", \"laws\": " + str(line["laws"]) + "}\n"
        outf.write(new_line)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result1", type=str)
    parser.add_argument("--result2", type=str)
    args = parser.parse_args()

    merge_result(args.result1, args.result2)

