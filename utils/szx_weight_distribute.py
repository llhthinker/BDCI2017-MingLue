import pickle
def func1(filename_list):
    laws_dict = {}
    for i in range(452):
        laws_dict[i] = 1
    for filename in filename_list:
        for line in open(filename, 'r', encoding='utf-8'):
            a = line.split('\t')
            laws = a[3].strip().split(',')
            for l in laws:
                laws_dict[int(l)-1] += 1
    count = 0.0
    for law in laws_dict:
        count += laws_dict[law]
    print(count)
    result_List = []
    for i in range(452):
        result_List.append(float(laws_dict[i])/count)
    print(result_List)
    # laws = sorted(laws_dict.items(), key=lambda d: d[1], reverse=True)
    with open('../pickles/weight_distribute.pkl', 'wb')as f:
        pickle.dump(result_List, f)

func1(['/Users/zxsong/Documents/BDCI2017-minglue-Semi/train.txt'])