# BDCI2017-MingLue
BDCI2017，让AI当法官：http://www.datafountain.cn/#/competitions/277/intro

## 任务说明
- Task 1: 罚金等级预测
- Task 2: 法条预测
## 代码文件夹说明
- **preprocessor: 存放数据预处理相关的代码**
    - builddataset.py: 将文本数据转化为索引数值，用于Task 1
    - buildmultidataset.py: 将文本数据转化为索引数值，用于Task 2
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
    - trainword2vecmodel.py: 根据训练集[train-file]和测试集[test-file]生成[word2vec-model]，用于pretrain
    ```
    python ./trainword2vecmodel.py --word2vec-model-path [word2vec-model] --train-file [train-file] --test-file [test-file]
    ```

- **utils: 存放一些工具类的代码**
    - calculatescore.py: 计算得分Micro-Averaged F1(Task 1)和Jaccard(Task 2)
    - statisticdata.py: 对数据进行一些统计分析
    - trainhelper.py: 训练需要的一些函数
    - multitrainhelper.py: Task 2训练需要的一些函数

- **models: 各种DL模型代码**

    | model\_id  | code\_file\_name   | model\_name   |
    | --------   | -----:  | :----:  |
    | 0     | fasttext.py |   FastText     |
    | 1     |   textcnn.py   |   TextCNN   |
    | 2     |    textrcnn.py    |  TextRCNN  |
    | 3     | (待补充) |   (待补充:RNN)     |
    | 4     |   hierarchical.py   |   HAN   |
    | 5     |    cnnwithdoc2vec.py    |  CNNWithDoc2Vec  |
    | 6     | rcnnwithdoc2vec.py |   RCNNWithDoc2Vec     |
    | ...   |   ...  |   ...   |

- **data: 将数据包装成pytorch中的Dataset**
    - mingluedata.py: **Task 1**
    - mingluemultidata.py: **Task 2**

- **notebooks: 用jupyter notebook做一些实验**

- **主目录: 存放训练和预测运行代码和相关配置代码**
    - train.py: Task 1 训练脚本
    ```
    python ./train.py --model-id [model\_id] --is-save [y/n]
    ```
    - multitrain.py: Task 2 训练脚本
    ```
    python ./multitrain.py --model-id [model\_id] --is-save [y/n]
    ```
    - predict.py: 预测脚本，载入已有模型进行预测并生成json格式的结果文件
    ```
    # 注意model_id要和model-path对应的Model保持一致
    python ./predict --task1-model-id [model\_id] --task1-model-path [model\_path] --task2-model-id [model\_id] --task2-model-path [model_path]
    ```
    - mix\_predict.py: 融合多个模型进行预测，目前只实现Task 1
    - config.py: 配置文件，其中Config类对应Task 1, MultiConfig类对应Task 2

## 数据文件夹说明
- **corpus: 存放训练数据和测试数据**
- **pickles: 存放pickle类型数据，包括：**
    - index2word.[*.]pkl
    - word2index.[*.]pkl
    - 保存的模型数据:
        - \*.[model\_name]表示Task 1的模型文件, 如params.pkl.1511507513.TextCNN
        - \*.multi.[model\_name]表示Task 2的模型文件,如params.pkl.1511514902.multi.TextCNN
    - ...
    
- **word2vec: 存放pre-train word embedding相关数据**
- **results: 存放预测结果文件(json)**