import numpy as np

import torch 
from torch.autograd import Variable

from models.fasttext import FastText
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN
from models.textrnn import TextRNN
from models.hierarchical import HAN
from models.cnnwithdoc2vec import CNNWithDoc2Vec
from models.rcnnwithdoc2vec import RCNNWithDoc2Vec
from models.modelwithelement import ModelWithElement
from models.CNNInception import CNNwithInception
import preprocessor.builddataset as bd
import preprocessor.getdoc2vec as gdv

import utils.statisticsdata as sd
import utils.calculatescore as cs


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.array(predictions) == np.array(labels))
            / len(labels))

def model_selector(config, model_id, use_element):
    model = None
    if use_element:
        print("use element")
        model = ModelWithElement(config, model_id)
    elif model_id == 0:
        model = FastText(config)
    elif model_id == 1:
        model = TextCNN(config)
    elif model_id == 2:
        model = TextRCNN(config)
    elif model_id == 3:
        model = TextRNN(config)
    elif model_id == 4:
        model = HAN(config)
    elif model_id == 5:
        model = CNNWithDoc2Vec(config)
    elif model_id == 6:
        model = RCNNWithDoc2Vec(config)
    elif model_id == 7:
        model = CNNwithInception(config)
    else:
        print("Input ERROR!")
        exit(2)
    return model


def _get_loss_weight(predicted, label, num_class):

    sample_per_class = torch.zeros(num_class)
    error_per_class = torch.zeros(num_class)
    for p, t in zip(predicted, label):
        # print(p, t)
        sample_per_class[t] += 1
        if p != t:
            error_per_class[t] += 1

    return error_per_class / sample_per_class


def do_eval(valid_loader, model, model_id, has_cuda, dmpv_model=None, dbow_model=None):
    """ 在验证集上做验证，报告损失、精确度"""
    true_labels = []
    predicted_labels = []
    model.is_training = False
    model.dropout_rate = 0
    for data in valid_loader:
        ids, texts, labels = data
        if has_cuda:
            texts = texts.cuda()
        if dmpv_model is not None and dbow_model is not None:  # cnn and rcnn with doc2vec
            doc2vec = gdv.build_doc2vec(ids, dmpv_model, dbow_model)
            if has_cuda:
                doc2vec = Variable(torch.FloatTensor(doc2vec).cuda())
            else:
                doc2vec = Variable(torch.FloatTensor(doc2vec))
            outputs = model(Variable(text), doc2vec)
        else:
            outputs = model(Variable(texts))
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels)
        predicted_labels.extend(predicted.cpu())

    loss_weight = _get_loss_weight(predicted_labels, true_labels, 8)
    print(true_labels[:10])
    print(predicted_labels[:10])
    print("Acc:", accuracy(predicted_labels, true_labels))
    score = cs.micro_avg_f1(predicted_labels, true_labels, model.num_class)
    print("Micro-Averaged F1:", score)
    model.is_training = True
    model.dropout_rate = 0.5
    return loss_weight, score


def build_element_vec(ids, all_element_vec):
    element_vec = []
    for id in ids:
        element_vec.append(all_element_vec[id])
    
    return np.array(element_vec, dtype=np.int64)
