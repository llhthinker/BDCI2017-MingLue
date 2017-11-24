import pickle
from gensim.models import Word2Vec
import numpy as np
import argparse

def save_dict(dic, fpath):
    output = open(fpath, 'wb')
    pickle.dump(dic, output)

def load_pickle(fpath):
    pkl_f = open(fpath, 'rb')
    return pickle.load(pkl_f)

def build_pretrain_emb(index2word, model_path, embedding_path):
    pretrain_emb = []
    model = Word2Vec.load(model_path)   
    vocab_size = len(index2word)
    print("index2word len:", vocab_size)
    embedding_size = model.layer1_size
    print("embedding size:", embedding_size)
    count = 0
    for i in range(vocab_size):
        word = index2word[i]
        if word in model.wv:
            pretrain_emb.append(model.wv[word])
        else:
            count += 1
            rand_emb = np.random.uniform(low=-1, high=1, size=embedding_size)
            pretrain_emb.append(rand_emb)
    print("rand init count:",count)
    pretrain_emb = np.array(pretrain_emb)
    np.save(embedding_path, pretrain_emb)

if __name__ == "__main__":

    # index2word_path = './pickles/index2word.pkl'
    # model_path = './word2vec/MingLueData.nr.word2vec.128d.model'
    # embedding_path = './word2vec/pretrain_emb.nr.128d.npy'

    parser = argparse.ArgumentParser()
    parser.add_argument("--index2word-path", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--embedding-path", type=str)
    args = parser.parse_args()

    index2word = load_pickle(args.index2word_path)
    build_pretrain_emb(index2word, args.model_path, args.embedding_path)
