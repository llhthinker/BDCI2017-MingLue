#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
import time 
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import preprocessor.buildmultidataset as bmd
import preprocessor.buildpretrainemb as bpe
import preprocessor.getdoc2vec as gdv

import utils.statisticsdata as sd

import utils.calculatescore as cs
from utils.trainhelper import model_selector
from utils.multitrainhelper import get_multi_label_from_output, where_result_reshape, do_eval
from data.mingluemultidata import MingLueMultiData

from config import MultiConfig

def main(model_id, is_save):
    config = MultiConfig()

    print("loading data...")
    ids, data, labels = bmd.load_data(config.data_path)
#    sd.show_text_len_distribution(data)
#    sd.show_label_text_len_distribution(labels, data)
    total_vocab_size = sd.count_vocab_size(data)
    print("total vocab size", total_vocab_size)
    force = config.force_word2index 
    if not force and os.path.exists(config.index2word_path) and os.path.exists(config.word2index_path):
        print("load word2index")
        dict_word2index = bpe.load_pickle(config.word2index_path)
    else:
        print("save word2index and index2word")
        count, dict_word2index, dict_index2word = bmd.build_vocabulary(data, min_count=config.min_count)
        bpe.save_dict(dict_index2word, config.index2word_path)
        bpe.save_dict(dict_word2index, config.word2index_path)
        return 
    if model_id == 4:
        train_data, train_labels = bmd.build_data_set_HAN(data, labels, dict_word2index, num_sentences=config.num_sentences,
                                                         sequence_length=config.sequence_length)
        train_ids, valid_ids = bmd.split_data(ids, radio=0.8)
        train_X, valid_X = bmd.split_data(train_data, radio=0.8)
        train_y, valid_y = bmd.split_data(train_labels, radio=0.8)

    else:
        train_ids, valid_ids = bmd.split_data(ids, radio=0.8)
        train_data, valid_data = bmd.split_data(data, radio=0.8)
        train_labels, valid_labels = bmd.split_data(labels, radio=0.8)
        

        train_ids, train_X, train_y = bmd.build_dataset(train_ids, train_data, train_labels,
                 dict_word2index, config.max_text_len, config.num_class)
        valid_ids, valid_X, valid_y = bmd.build_dataset(valid_ids, valid_data, valid_labels, 
        dict_word2index, config.max_text_len, config.num_class)
    print("trainset size:", len(train_ids))
    print("validset size:", len(valid_ids))

#    train_ids, train_X, train_y = bd.over_sample(train_ids, train_X, train_y)
#    print(train_y.shape[0], Counter(train_y))
    if is_save == 'y':
        # all_train_ids, all_train_X, all_train_y = bmd.build_dataset(ids, data, labels,
        #     dict_word2index, config.max_text_len, config.num_class)
        # dataset = MingLueMultiData(all_train_ids, all_train_X, all_train_y)
        dataset = MingLueMultiData(valid_ids, valid_X, valid_y)
    else: 
        dataset = MingLueMultiData(train_ids, train_X, train_y)
    batch_size = config.batch_size
    if model_id == 4:
        batch_size = config.han_batch_size
    train_loader = DataLoader(dataset=dataset, 
                               batch_size=batch_size, # 更改便于为不同模型传递不同batch
                               shuffle=True,
                               num_workers=config.num_workers)
    dataset = MingLueMultiData(valid_ids, valid_X, valid_y)
    valid_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size, # 更改便于为不同模型传递不同batch
                              shuffle=False,
                              num_workers=config.num_workers)
    if model_id == 5 or model_id == 6:  # cnn and rcnn with doc2vec
        dmpv_model, dbow_model = gdv.load_doc2vec_model(config.dmpv_model_path, config.dbow_model_path)
    print("data loaded")
 
    config.vocab_size = len(dict_word2index)
    print('config vocab size:', config.vocab_size)
    model = model_selector(config, model_id)
    if config.has_cuda:
        model = model.cuda()  
    
    loss_fun = nn.MultiLabelSoftMarginLoss()
    
#    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = model.get_optimizer(config.learning_rate,
                                    config.learning_rate2,
                                    config.weight_decay)
    print("training...")

    weight_count = 0
    for epoch in range(config.epoch_num):
        print("lr:",config.learning_rate,"lr2:",config.learning_rate2)
        running_loss = 0.0
        running_jaccard = 0.0
        for i, data in enumerate(train_loader, 0):
            ids, texts, labels = data
            # TODO
            if model_id == 4:
                pass
            if config.has_cuda:    
                inputs, labels = Variable(texts.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(texts), Variable(labels)
            optimizer.zero_grad()
            if model_id == 5 or model_id == 6:  # cnn and rcnn with doc2vec
                doc2vec = gdv.build_doc2vec(ids, dmpv_model, dbow_model)
                if config.has_cuda:
                    doc2vec = Variable(torch.FloatTensor(doc2vec).cuda())
                else:
                    doc2vec = Variable(torch.FloatTensor(doc2vec))
                # [batch_size, (doc2vec_size*2)]
                # print(doc2vec.size())
                outputs = model(inputs, doc2vec)
            
            else:
                outputs = model(inputs)
            loss = loss_fun(outputs, labels.float())  # or weight *labels.float() 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
       

            if i % config.step == config.step-1:
                if epoch % config.epoch_step == config.epoch_step-1:
                    predicted_labels = get_multi_label_from_output(outputs, config)
                    
                    true_label = labels.data.cpu().numpy()
                    rows, true_label = np.where(true_label == 1)
                    true_label = where_result_reshape(outputs.size()[0], rows, true_label)

                    running_jaccard = cs.jaccard(predicted_labels, true_label)
                    print('[%d, %5d] loss: %.3f, jaccard: %.3f' %
                        (epoch + 1, i + 1, running_loss / config.step, running_jaccard))
                running_loss = 0.0

    
        if is_save != 'y' and epoch % config.epoch_step == config.epoch_step-1:
            print("predicting...")
            if model_id == 5 or model_id == 6:
                loss_weight = do_eval(valid_loader, model, model_id, config, dmpv_model, dbow_model)
            else:
                loss_weight = do_eval(valid_loader, model, model_id, config)
            if epoch >= 3:
                weight_count += 1
            #    total_loss_weight += loss_weight
            #    print("avg_loss_weight:",total_loss_weight/weight_count)    

        if epoch >= config.begin_epoch-1:
            if epoch >= config.begin_epoch and config.learning_rate2 == 0:
                config.learning_rate2 = 2e-4
            elif config.learning_rate2 > 0:
                config.learning_rate2 *= config.lr_decay
                if config.learning_rate2 <= 1e-5:
                    config.learning_rate2 = 1e-5
            config.learning_rate = config.learning_rate * config.lr_decay
            optimizer = model.get_optimizer(config.learning_rate,
                                            config.learning_rate2,
                                            config.weight_decay)
    time_stamp = str(int(time.time()))

    
    if is_save == "y":
        torch.save(model.state_dict(), config.model_path+"."+time_stamp+".multi."+config.model_names[model_id])
    else:
        print("not save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--is-save", type=str)
    args = parser.parse_args()

    main(args.model_id, args.is_save)
