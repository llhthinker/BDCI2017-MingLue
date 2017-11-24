# BDCI2017-MingLue
BDCI2017，让AI当法官：http://www.datafountain.cn/#/competitions/277/intro

## 代码文件夹
- preprocessor: 存放数据预处理相关的代码
    - builddataset.py: 将文本数据转化为索引数值，用于Task 1, 罚金等级预测
    - buildmultidataset.py: 将文本数据转化为索引数值，用于Task 2, 法条预测
    - sampledata.py: 对文本进行采样：从原始文本从取前count个样本
    ```
    python ./sampledata.py -i [input-file-path] -o [output-file-path] -c [count]
    ```
    - segtext.py: 分词
    ```
    python ./segtext.py -i [input-file-path] -o [output-file-path]
    ```
    - shuffledata.py: 将文本按行随机打乱
    ```
    python ./shuffledata.py -i [input-file-path] -o [output-file-path]
    ```
    - trainword2vecmodel.py: 根据训练集[train-file]和测试集[test-file]生成word2vec-model，用于pretrain
    ```
    python ./trainword2vecmodel.py --word2vec-model-path [word2vec-model] --train-file [train-file] --test-file [test-file]
    ```

- utils: 存放一些工具类的代码
    - calculatescore.py: 计算得分Micro-Averaged F1(Task 1)和Jaccard(Task 2)
    - statisticdata.py: 对数据进行一些统计分析

- models
- data
- notebooks

## 数据文件夹
- corpus
- pickles
- word2vec
- results