import numpy as np

import torch 
from torch.autograd import Variable

import preprocessor.getdoc2vec as gdv

import utils.statisticsdata as sd
import utils.calculatescore as cs


def where_result_reshape(row_size, rows, indices):
    rows = rows.tolist()
    indices = indices.tolist()
    result = []
    for i in range(row_size):
        result.append([])
        for ri, row in enumerate(rows):
            if row > i:
                break
            if i == row:
                result[row].append(indices[ri])
    return result


def get_multi_label_from_output(outputs, config):
    predicted = torch.sigmoid(outputs)
    predicted = predicted.data.numpy()
    rows, predicted = np.where(predicted > config.max_prob)
    predicted_labels = where_result_reshape(outputs.size()[0], rows, predicted)
    
    _, predicted = torch.max(outputs.data, 1)
    predicted_max_labels = predicted.numpy().tolist()
    predicted_labels_len = len(predicted_labels)
    # print(predicted_labels_len, len(predicted_max_labels))
    if predicted_labels_len != len(predicted_max_labels):
        raise ValueError("predicted_labels'length should equal predicted_max_labels' length")
    for pl_i in range(len(predicted_labels)):
        predicted_labels[pl_i].extend(predicted_max_labels[pl_i])

    return predicted_labels


def do_eval(valid_loader, model, model_id, config, dmpv_model=None, dbow_model=None):
    """ 在验证集上做验证，报告损失、精确度"""
    model.is_training = False
    true_labels = []
    predicted_labels = []
    
    for data in valid_loader:
        ids, texts, labels = data
        if config.has_cuda:
            texts = texts.cuda()
        if dmpv_model is not None and dbow_model is not None:  # cnn and rcnn with doc2vec
            doc2vec = gdv.build_doc2vec(ids, dmpv_model, dbow_model)
            if config.has_cuda:
                doc2vec = Variable(torch.FloatTensor(doc2vec).cuda())
            else:
                doc2vec = Variable(torch.FloatTensor(doc2vec))
            outputs = model(Variable(texts), doc2vec)
        else:
            outputs = model(Variable(texts))

        predicted_labels.extend(get_multi_label_from_output(outputs, config))
        true_label = labels.numpy()
        rows, true_label = np.where(true_label == 1)
        true_labels.extend(where_result_reshape(outputs.size()[0], rows, true_label))
 #   loss_weight = get_loss_weight(predicted_labels, true_labels, 8)
    print(true_labels[:3])
    print(predicted_labels[:3])
    print("Jaccard:", cs.jaccard(predicted_labels, true_labels))
    model.is_training = True
    return None