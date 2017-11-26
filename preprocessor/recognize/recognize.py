import re
import json
import toolkit

def recognize_date(text):
    """Recognize all dates (日期) with [date].
    """
    num = r'[\d零〇一二三四五六七八九十同]'
    regex_pattern = r'({0}+年)?{0}+月({0}+日)?({0}+时)?({0}+分)?许?'.format(num)
    return re.sub(regex_pattern, '[date]', text)

def _testdrive_recognize_date():
    text = ['于2016年12月17日出生于',
            '于2016年12月17日18时许持刀抢劫',
            '于一九九六年一月十二日因盗窃']
    for t in text:
        print(t + '\t' + recognize_date(t))


def recognize_BAC(text, category_dict):
    """Recognize and discrete all blood alcohol concentration (血液酒精浓度).
    """
    def discrete_BAC(MatchObject):
        """Discrete the BAC according to the range.
        """
        BAC = float(MatchObject.group(2))
        if MatchObject.group(3) == '':
            BAC = BAC * 100
        for category in category_dict:
            range = category_dict[category]
            if BAC >= range[0] and BAC < range[1]:
                return category

    regex_pattern = r'((\d+\.?\d*)mg\/(100|)ml)'
    return re.sub(regex_pattern, discrete_BAC, text)

def _testdrive_recognize_BAC():
    text = ['被告人汪某某血样中酒精含量为117.4mg/100ml，属醉酒驾驶。',
            '被告人汪某某血样中酒精含量为72mg/100ml，属饮酒驾驶。',
            '被告人汪某某血样中酒精含量为1.174mg/ml，属醉酒驾驶。']
    with open('./category_BAC.json', 'r') as f:
        category_dict = json.loads(f.read())
    for t in text:
        print(t + '\t' + recognize_BAC(t, category_dict))


def recognize_weight(text, category_dict):
    """Recognize and discrete all weight (重量).
    """
    def discrete_weight(MatchObject):
        """Discrete the matched weight according to the range.
        """
        weight = float(MatchObject.group(2))
        if MatchObject.group(3) == '千克':
            weight *= 1e3
        elif MatchObject.group(3) == '吨':
            weight *= 1e6
        for category in category_dict:
            range = category_dict[category]
            if weight >= range[0] and weight < range[1]:
                return category

    regex_pattern = r'((\d+\.?\d*)[余多]?(克|千克|吨))'
    return re.sub(regex_pattern, discrete_weight, text)

def _testdrive_recognize_weight():
    text = ['一车载有22840千克的玉米过好磅',
            '卖了190多吨玉米',
            '该小包疑似甲基苯丙胺重15.12克']
    with open('./category_weight.json', 'r') as f:
        category_dict = json.loads(f.read())
    for t in text:
        print(t + '\t' + recognize_weight(t, category_dict))


def recognize_money(text, category_dict):
    """Recognize and discrete all money (金钱).
    """
    def discrete_money(MatchObject):
        """Discrete the matched money according to the range.
        """
        if MatchObject.group(1) != None:
            # 说明是阿拉伯数字和中文数字结合型。
            # 在该型中，中文数字识别为阿拉伯数字的数量级，需相乘。
            magnitude = {'': 1, '十': 10, '百': 1e2, '千': 1e3, '万': 1e4,
                         '十万': 1e5, '百万': 1e6, '千万': 1e7, '亿': 1e8}
            base = float(re.sub(r'[,，]', '', MatchObject.group(2)))
            money = base * magnitude[MatchObject.group(3)]
        elif MatchObject.group(4) != None:
            # 说明是纯中文数字型。
            # 在该型中，中文数字识别为阿拉伯数字的中文写法，需转化。
            money = float(toolkit.zhnum2int(MatchObject.group(5)))
        for category in category_dict:
            range = category_dict[category]
            if money >= range[0] and money < range[1]:
                return category

    # 中文数字情况比较复杂，大致可以分为两类：
    # 1）阿拉伯数字和中文数字结合型
    # 在该型中，中文数字为阿拉伯数字的数量级，需相乘。如：95.8万元。
    # 2）纯中文数字型
    # 在该型中，中文数字为阿拉伯数字的中文写法，需转化。如：一千元。
    regex_pattern = r'((\d+[,，]?\d*\.?\d*)(|十|百|千|万|十万|百万|千万|亿)?[余多]?(?:元|块钱))|' + \
                    r'(([零一二两三四五六七八九十百千万亿]+)[余多]?(?:元|块钱))'
    return re.sub(regex_pattern, discrete_money, text)

def _testdrive_recognize_money():
    text = ['盗窃两千七百块钱。',
            '盗窃十一万元。',
            '盗窃28万元。',
            '盗窃1千元。',
            '盗窃一千元。',
            '盗窃95.8万元。',
            '盗窃2百万元。',
            '盗窃4，200元。']
    with open('./category_money.json', 'r') as f:
        category_dict = json.loads(f.read())
    for t in text:
        print(t + '\t' + recognize_money(t, category_dict))


if __name__ == '__main__':
    _testdrive_recognize_weight()
