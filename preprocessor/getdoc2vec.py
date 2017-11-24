from gensim.models import Doc2Vec, doc2vec
import numpy as np

def load_doc2vec_model(dmpv_model_path, dbow_model_path):
    dmpv_model = Doc2Vec.load(dmpv_model_path)
    dbow_model = Doc2Vec.load(dbow_model_path)
    return dmpv_model, dbow_model


def build_doc2vec(ids, dmpv_model, dbow_model):
    batch_doc2vecs = []
    for id in ids:
        dmpv_vec = dmpv_model.docvecs[id]
        dbow_vec = dbow_model.docvecs[id]
        vec = np.concatenate([dmpv_vec, dbow_vec])
        batch_doc2vecs.append(vec)
    
    return np.array(batch_doc2vecs)

