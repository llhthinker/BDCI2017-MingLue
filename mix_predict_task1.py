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


def load_model(model_path, model_id, config):
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
#    final_model_path = config.model_path+"."+time_stamp+"."+config.model_names[model_id]
    print("load model data:", model_path)
    model.load_state_dict(torch.load(model_path))
    if config.has_cuda:
        model = model.cuda()
    return model


def predict(test_loader, test_loader_HAN, rcnn_model, han_model, config):

    predicted_labels = []
    for data, han_data in zip(test_loader, test_loader_HAN):
        texts, han_texts = data, han_data
        if config.has_cuda:
            texts = texts.cuda()
            han_texts = han_texts.cuda()
        rcnn_outputs = rcnn_model(Variable(texts))
        han_outputs = han_model(Variable(han_texts))

        total_output = rcnn_outputs + han_outputs
        _, predicted = torch.max(total_output.data, 1)
        predicted = predicted.cpu().numpy().tolist()
        # predicted = [i[0] for i in predicted]
        predicted_labels.extend(predicted)

    predicted_labels = [label+1 for label in predicted_labels]

    return predicted_labels


def generate_result_json(tests_id, predicted_labels, result_path):
    test_len = len(tests_id)
    tmp_law = [1]
    time_stamp = str(int(time.time()))
    outf = open(result_path+"."+time_stamp, 'a')
    for i in range(test_len):
        result = {}
        result["id"] = tests_id[i]
        result["penalty"] = predicted_labels[i]
        result["laws"] = [-1]
      #  res.append(result)
        json.dump(result, outf)
        outf.write('\n')


def main(rcnn_model_path, han_model_path):
    config = Config()
    config.is_training = False
    config.dropout_rate = 0.0

    print("loading data...")
    dict_word2index = bpe.load_pickle(config.word2index_path)
    tests_id, test_data = bd.load_test_data(config.test_path)
    test_X = bd.build_test_data(test_data, dict_word2index, config.max_text_len)

    testset = MingLueTestData(test_X)
    test_loader = DataLoader(dataset=testset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)

    test_X_HAN = bd.build_test_data_HAN(test_data, dict_word2index, config.num_sentences, config.sequence_length)

    testset = MingLueTestData(test_X_HAN)
    test_loader_HAN = DataLoader(dataset=testset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)

    
    config.vocab_size = len(dict_word2index)
    print("loading model...")

    rcnn_model = load_model(rcnn_model_path, 2, config)
    han_model = load_model(han_model_path, 4, config)
    print("model loaded")

    print("predicting...")
    predicted_labels = predict(test_loader, test_loader_HAN, rcnn_model, han_model,  config)

    generate_result_json(tests_id, predicted_labels, config.result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rcnn-model-path", type=str)
    parser.add_argument("--han-model-path", type=str)


    args = parser.parse_args()

    main(args.rcnn_model_path, args.han_model_path)
#    test()



