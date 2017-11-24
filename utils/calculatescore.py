""" calculate score: 
1. Micro-Averaged F1
2. Jaccard 
"""

from collections import Counter


def micro_avg_f1(predict_label, true_label, label_size):
    N = len(predict_label)
    m = label_size
    w = Counter(true_label)
    print(w)
    score = 0
    for i in range(m):
        score += w[i] * f1(predict_label, true_label, i)

    return score / float(N)


def f1(predict_label, true_label, cur_label):
    true_pos, false_pos = 0, 0
    false_neg = 0
    for i in range(len(predict_label)):
        if predict_label[i] == cur_label:
            if true_label[i] == cur_label:
                true_pos += 1
            else:
                false_pos += 1
        else:  # predict_label != cur_label
            if true_label[i] == cur_label:
                false_neg += 1
    if true_pos == 0:
        precision, recall = 0, 0
    else:
        precision = true_pos / float(true_pos + false_pos)
        recall = true_pos / float(true_pos + false_neg)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def jaccard(predicted_label, true_label):
    # print("predicted labels: ", predicted_label)
    # print("true labels: ", true_label)
    p = 0
    N = len(predicted_label)
    predict_set_size = 0
    true_set_size = 0
    for i in range(N):
        Li = set(true_label[i])
        Lig = set(predicted_label[i])
        p += len(Li & Lig) / len(Li | Lig)
        
        true_set_size += len(Li)
        predict_set_size += len(Lig)

    print("predict_set_size / true_set_size: ", predict_set_size / true_set_size)
    return p / N


def test():
    predict_label = [0, 0, 1, 2, 3, 2, 5, 6, 6, 7, 1, 1]
    true_label = [0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 1, 2]
    label_size = 8
    print("Micro-Averaged F1: ", micro_avg_f1(predict_label, true_label, label_size))


if __name__ == "__main__":
    test()
