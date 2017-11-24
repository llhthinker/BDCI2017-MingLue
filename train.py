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

import preprocessor.builddataset as bd
import preprocessor.buildpretrainemb as bpe
import preprocessor.getdoc2vec as gdv

import utils.statisticsdata as sd
import utils.calculatescore as cs
from utils.trainhelper import accuracy, model_selector, do_eval

from config import Config

from data.mingluedata import MingLueData


def main(model_id, is_save):
    config = Config()
   # model_id = int(input("Please select a model(input model id):\n0: fastText\n1: TextCNN\n2: TextRCNN\n4: HAN\nInput: "))
   # is_save = input("Save Model?(y/n): ")
    print("loading data...")
    ids, data, labels = bd.load_data(config.data_path)
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
        count, dict_word2index, dict_index2word = bd.build_vocabulary(data, min_count=config.min_count)
        bpe.save_dict(dict_index2word, config.index2word_path)
        bpe.save_dict(dict_word2index, config.word2index_path)
        return 
    if model_id == 4:
        train_data, train_labels = bd.build_data_set_HAN(data, labels, dict_word2index, num_sentences=config.num_sentences,
                                                         sequence_length=config.sequence_length)
        train_ids, valid_ids = bd.split_data(ids, radio=0.8)
        train_X, valid_X = bd.split_data(train_data, radio=0.8)
        train_y, valid_y = bd.split_data(train_labels, radio=0.8)

    else:
        train_ids, valid_ids = bd.split_data(ids, radio=0.8)
        train_data, valid_data = bd.split_data(data, radio=0.8)
        train_labels, valid_labels = bd.split_data(labels, radio=0.8)
        # over sample for train data
        train_ids, train_X, train_y = bd.build_dataset_over_sample(train_ids, train_data, 
                                        train_labels, dict_word2index, config.max_text_len)
        valid_ids, valid_X, valid_y = bd.build_dataset(valid_ids, valid_data, 
                        valid_labels, dict_word2index, config.max_text_len)
        print("trainset size:", len(train_ids))
        print("validset size:", len(valid_ids))

#    train_ids, train_X, train_y = bd.over_sample(train_ids, train_X, train_y)
#    print(train_y.shape[0], Counter(train_y))
    if is_save == 'y':
        all_train_ids, all_train_X, all_train_y = bd.build_dataset(ids, data, 
                                    labels, dict_word2index, config.max_text_len)
        dataset = MingLueData(all_train_ids, all_train_X, all_train_y)
    else: 
        dataset = MingLueData(train_ids, train_X, train_y)
    batch_size = config.batch_size
    if model_id == 4:
        batch_size = config.han_batch_size
    train_loader = DataLoader(dataset=dataset, 
                               batch_size=batch_size, # 更改便于为不同模型传递不同batch
                               shuffle=True,
                               num_workers=config.num_workers)
    dataset = MingLueData(valid_ids, valid_X, valid_y)
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
    

    loss_weight = torch.FloatTensor(config.loss_weight_value)
    loss_weight = loss_weight + 1 - loss_weight.mean()
    print("loss weight:",loss_weight)
#    loss_fun = nn.CrossEntropyLoss(loss_weight.cuda())
    
    loss_fun = nn.CrossEntropyLoss()
#    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = model.get_optimizer(config.learning_rate,
                                    config.learning_rate2,
                                    config.weight_decay)
    print("training...")

    weight_count = 0
    total_loss_weight = torch.FloatTensor(torch.zeros(8))
    for epoch in range(config.epoch_num):
        print("lr:",config.learning_rate,"lr2:",config.learning_rate2)
        running_loss = 0.0
        running_acc = 0.0
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
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
       

            if i % config.step == config.step-1:
                if epoch % config.epoch_step == config.epoch_step-1:
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu().numpy().tolist()
                    predicted = [i[0] for i in predicted]
                    running_acc = accuracy(predicted, labels.data.cpu().numpy())
                    print('[%d, %5d] loss: %.3f, acc: %.3f' %
                        (epoch + 1, i + 1, running_loss / config.step, running_acc))
                running_loss = 0.0
    
        if is_save != 'y' and epoch % config.epoch_step == config.epoch_step-1:
            print("predicting...")
            if model_id == 5 or model_id == 6:
                loss_weight = do_eval(valid_loader, model, model_id,
                                     config.has_cuda, dmpv_model, dbow_model)
            else:
                loss_weight = do_eval(valid_loader, model, model_id, config.has_cuda)
            if epoch >= 3:
                weight_count += 1
                total_loss_weight += loss_weight
                print("avg_loss_weight:",total_loss_weight/weight_count)    

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
        torch.save(model.state_dict(), config.model_path+"."+time_stamp+"."+config.model_names[model_id])
    else:
        print("not save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int)
    parser.add_argument("--is-save", type=str)
    args = parser.parse_args()

    main(args.model_id, args.is_save)
