import re
import pickle

numdict = {'零':0, '一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10, '百':100, '千':1000, '万':10000,
        '〇':0, '亿':100000000,
        '壹':1, '贰':2, '叁':3, '肆':4, '伍':5, '陆':6, '柒':7, '捌':8, '玖':9, '拾':10, '佰':100, '仟':1000, '萬':10000,
       }

def zhnum2int(a):
  count = 0
  result = 0
  tmp = 0
  Billion = 0
  while count < len(a):
    tmpChr = a[count]
    #print tmpChr
    tmpNum = numdict.get(tmpChr, None)
    #如果等于1亿
    if tmpNum == 100000000:
      result = result + tmp
      result = result * tmpNum
      #获得亿以上的数量，将其保存在中间变量Billion中并清空result
      Billion = Billion * 100000000 + result
      result = 0
      tmp = 0
    #如果等于1万
    elif tmpNum == 10000:
      result = result + tmp
      result = result * tmpNum
      tmp = 0
    #如果等于十或者百，千
    elif tmpNum >= 10:
      if tmp == 0:
        tmp = 1
      result = result + tmpNum * tmp
      tmp = 0
    #如果是个位数
    elif tmpNum is not None:
      tmp = tmp * 10 + tmpNum
    count += 1
  result = result + tmp
  result = result + Billion
  return result

def extract_laws(filename):
    '''

    :param filename: 文本
    :return: dict
    '''
    i = 0
    text = []
    for line in open(filename, "r", encoding="utf-8"):
        text.append(line.strip())
        i += 1
    result = {}
    # answer = {}
    for t in text:
        sline = t.split("\t")
        zhnum = []
        tmp = []
        # obgj = re.findall('《中(.*?)法》', t)
        # if obgj:
        #     print(obgj)
        # continue
        obgj = re.findall('《.*?》(.*?)。', t)
        if obgj:
            for a in obgj:
                r1 = re.compile('第(.*?)条')
                tmp.extend(r1.findall(a))
                r2 = re.compile('、第(.*?)条')
                tmp.extend(r2.findall(a))
                r3 = re.compile('，第(.*?)条')
                tmp.extend(r3.findall(a))
                tmp = list(set(tmp))
            for l in tmp:
                if str(l).isdigit():
                    zhnum.append(int(l))
                    break
                flag = 1
                for ll in l:
                    if ll not in numdict:
                        flag = 0
                if flag == 1:
                    zhnum.append(zhnum2int(l))
            # if len(obgj) > 1:
            #     print(obgj)
            #     print(zhnum)
            result[str(sline[0])] = zhnum
            # answer[str(sline[0])] = sline[3].split(',')
        else:
            result[str(sline[0])] = []
            pass
    return result
    # count = 0
    # for k in result:
    #     for rk in result[k]:
    #         if str(rk) not in answer[k]:
    #             print(result[k], answer[k])
    #             count += 1
    #             break
    # print(count)
    # print(len(result))


if __name__ == "__main__":
    filename = "/Users/zxsong/Documents/BDCI2017-minglue-Semi/train.txt"
    result = extract_laws(filename)
    with open("../pickles/extract_laws.pkl", "wb")as f:
        pickle.dump(result, f)

    filename = "/Users/zxsong/Documents/BDCI2017-minglue-Semi/test.txt"
    result = extract_laws(filename)
    with open("../pickles/extract_laws_test.pkl", "wb")as f:
        pickle.dump(result, f)