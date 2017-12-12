import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import json
import time
import numpy as np
import argparse

from models.fasttext import FastText
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN
from models.hierarchical import HAN

from config import Config, MultiConfig
from data.mingluedata import MingLueTestData
import preprocessor.builddataset as bd
import preprocessor.buildpretrainemb as bpe

from utils.multitrainhelper import get_multi_label_from_output



def load_multi_model(model_path, model_id, config):
    if model_id == 0:
        model = FastText(config)
    elif model_id == 1:
        model = TextCNN(config)
    elif model_id == 2:
        model = TextRCNN(config)
    elif model_id == 4:
        model = HAN(config)
#    print(model)
#    time_stamp = '1510844987' 
#    final_model_path = config.model_path+"."+time_stamp+".multi."+config.model_names[model_id]
    print("load model data:", model_path)
    model.load_state_dict(torch.load(model_path))
    if config.has_cuda:
        model = model.cuda()
    
    return model


def predict_multi_label(test_loader, multi_model, config):

    predicted_multi_labels = []

    for data in test_loader:
        texts = data
        if config.has_cuda:
            texts = texts.cuda()
        outputs = multi_model(Variable(texts))
        predicted_multi_labels.extend(get_multi_label_from_output(outputs, config))

    new_predict_multi_labels = []
    for multi_label in predicted_multi_labels:
        new_predict_multi_labels.append([label+1 for label in multi_label])
    return new_predict_multi_labels


def generate_result_json(tests_id, predicted_multi_labels, result_path):
    test_len = len(tests_id)
    tmp_law = [1]
    time_stamp = str(int(time.time()))
    outf = open(result_path+"."+time_stamp, 'a')
    for i in range(test_len):
        result = {}
        result["id"] = tests_id[i]
        result["penalty"] = 1
        result["laws"] = list(set(predicted_multi_labels[i]))
        # result["laws"] = [-1]
      #  res.append(result)
        json.dump(result, outf)
        outf.write('\n')


def main(task2_model_id, task2_model_path):
    multi_config = MultiConfig()
    multi_config.is_training = False
    multi_config.dropout_rate = 0.0

    print("loading data...")
    dict_word2index = bpe.load_pickle(multi_config.word2index_path)
    tests_id, test_data = bd.load_test_data(multi_config.test_path)
    if task2_model_id != 4:
        test_X = bd.build_test_data(test_data, dict_word2index, multi_config.max_text_len)
    else:
        test_X = bd.build_test_data_HAN(test_data, dict_word2index, multi_config.num_sentences, multi_config.sequence_length)

    testset = MingLueTestData(test_X)
    test_loader = DataLoader(dataset=testset,
                             batch_size=multi_config.batch_size,
                             shuffle=False,
                             num_workers=multi_config.num_workers)
    
    multi_config.vocab_size = len(dict_word2index)
    print("loading model...")
    model2 = load_multi_model(task2_model_path, task2_model_id, multi_config)
    
    print("model loaded")

    print("predicting...")
    predicted_multi_labels = [[]]
    predicted_multi_labels = predict_multi_label(test_loader, model2, multi_config)
    generate_result_json(tests_id, predicted_multi_labels, multi_config.result_path)


def test():
    config = Config()
    tests_id = ["12", "432"]
    predicted_labels = [5,7]
    predicted_multi_labels = [[1,2],[4,3,6]]
    generate_result_json(tests_id, predicted_labels, predicted_multi_labels, config.result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task2-model-id", type=int)
    parser.add_argument("--task2-model-path", type=str)
    args = parser.parse_args()

    main(args.task2_model_id, args.task2_model_path)
#    test()
