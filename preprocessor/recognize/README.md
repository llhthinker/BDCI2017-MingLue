# 预处理程序说明

## 运行方式

`python preprocessing.py --input-file='../data/1-train/train.txt' --output-file='./train.txt'`
`python preprocessing.py --input-file='../data/2-test/test.txt' --output-file='./test.txt'`

以上两个参数`--input-file`和`--output-file`是必须要指定的。

可选参数如下：
人名：--nr 和 --no-nr
地名：--ns 和 --no-ns
数字：--m 和 --no-m
日期：--date 和 --no-date
金钱：--money 和 --no-money
重量：--weight 和 --no-weight
血液酒精浓度：--BAC 和 --no-BAC

左边一列表示识别、替换、离散化这些成分（True），右边相反（False）。
默认是全部是True。因此如果想要不识别某些成分，需要在命令后面加上`--no-xxx`，如：

`python preprocessing.py --input-file='../data/1-train/train.txt' --output-file='./train.txt' --no-nr --no-ns --no-m`

表示不要替换人名、地名和数字。

## 文件组织

主要有三个模块：

* preprocessing.py 主模块。是预处理的基本流程。
基本思路是：line_preprocessing()是对语料的每一行进行处理，然后在preprocessing()中采用multiprocessing.Pool和map函数手动将任务分配至所有CPU核上并行运行。之所以不用jieba内置的并行，是因为那个好像并行时内存也要翻倍，而且会有一个主进程限制速度，所以一般8核速度基本到顶。如果自己用map来实现并行，理论上可以多少核速度就翻几倍，且几乎不耗内存。

* recognize.py 包含了所有识别和离散化的函数。

* toolkit.py 用到的一些工具函数。主要是全角转半角、中文数字转阿拉伯数字等函数。

另含有三个离散化设置json格式文件：

* category_money.json

* category_weight.json

* category_BAC.json

里面是对金钱、重量、血液酒精浓度离散化的配置。是按照赵老师给的那篇论文设置的，可以自己修改。



## 速度

非常快，纯分词（不识别nr、ns、m）训练集跑了44.92s，测试集跑了33.70s。
如果要识别nr、ns、m，则需要用到词性标注，但也很快，训练集跑了191.90s，测试集跑了131.82s。
