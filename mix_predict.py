import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import json
import time

from models.fasttext import FastText
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN

from utils.config import Config
from data.mingluedata import MingLueTestData
import prepropressing.builddataset as bd


def load_model(model_id, config, time_stamp):
    if model_id == 0:
        model = FastText(config)
    elif model_id == 1:
        model = TextCNN(config)
    elif model_id == 2:
        model = TextRCNN(config)
    
   # time_stamp = '1509189841' 
    model.load_state_dict(torch.load(config.model_path+"."+time_stamp+"."+config.model_names[model_id]))

    return model.cuda()


def predict(test_loader, models,num_class):

    predicted_labels = []
    for data in test_loader:
        # print(data.size())
        total_output = torch.FloatTensor(torch.zeros(data.size()[0], num_class)).cuda()
        for model in models:
            texts = data
            outputs = model(Variable(texts.cuda()))
            # print(outputs.data)
            total_output += outputs.data
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted.cpu())

    predicted_labels = [label+1 for label in predicted_labels]

    return predicted_labels


def generate_result_json(tests_id, predicted_labels, result_path):
    test_len = len(tests_id)
    res = []
    tmp_law = [1]
    time_stamp = str(int(time.time()))
    outf = open(result_path+"."+time_stamp, 'a')
    for i in range(test_len):
        result = {}
        result["id"] = tests_id[i]
        result["penalty"] = predicted_labels[i]
        
        result["laws"] = [1]
      #  res.append(result)
        json.dump(result, outf)
        outf.write('\n')


def main():
    config = Config()
   # model_id = int(input("Please select a model(input model id):\n0: fastText\n1: TextCNN\n2: TextRCNN\nInput: "))

    print("loading model...")
    cnn_model = load_model(model_id=1, config=config, time_stamp='1508748727')
    rcnn_model = load_model(model_id=2, config=config, time_stamp='1509189841')

    print("model loaded")
    print("loading data...")
    data, labels = bd.load_data(config.data_path)
    count, dict_word2index, dict_index2word = bd.build_vocabulary(data,
                                             vocabulary_size=config.vocab_size)
    tests_id, test_data = bd.load_test_data(config.test_path)
    test_X = bd.build_test_data(test_data, dict_word2index, max_text_len=config.max_text_len)

    testset = MingLueTestData(test_X)
    test_loader = DataLoader(dataset=testset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)
    print("predicting...")
    predicted_labels = predict(test_loader, [cnn_model, rcnn_model], config.num_class)
    generate_result_json(tests_id, predicted_labels, config.result_path)


def test():
    config = Config()
    tests_id = ["12", "432"]
    predicted_labels = [5,7]
    generate_result_json(tests_id, predicted_labels, config.result_path)

if __name__ == "__main__":
    main()
#    test()
