import numpy as np
import scipy.sparse
from octis.models.gin_topic_model.utils.create_graph import SimGraph
from tqdm.notebook import tqdm
from gensim.models import Word2Vec
import multiprocessing as mp

def dgl_graph_from_list(texts, token2id, method='SimilarityGraph', emb_mat=None, eps_simGraph=None):
    graphs = []
    if method=='SimilarityGraph':
        simGraph = SimGraph(token2id, emb_mat, eps_simGraph)
        for text in tqdm(texts, desc='Creating Similarity graph', leave=False):
            graphs.append(simGraph._text_to_graph(text))

    else:
        raise Exception('Graph Method Not Implemented Error..!!')

    return graphs


def w2v_from_list(texts, vocab, save_path=None,
                  min_count=2, dim=300, epochs=50,
                  workers=mp.cpu_count(), negative_samples=10,
                  window_size=4):
    """
    :param text: list of  documents
    :param vocab: list of words
    """ 
    texts = [s.split() for s in texts]
    model = Word2Vec(texts, min_count=min_count, sg=1, vector_size=dim, epochs=epochs,
                     workers=workers, negative=negative_samples, window=window_size)
    
    embedding_matrix = np.zeros((len(vocab), dim), dtype=np.float64)
    for i,v in enumerate(vocab):
        missing_words=0
        try:
            embedding_matrix[i] = model.wv[v]
        except:
            missing_words += 1
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(dim,))
    
    print('Embeddings are not found for {} words'.format(missing_words))
    #saving the emb matrix
    if save_path:
        np.save(open(save_path, 'wb'), embedding_matrix)
    
    return embedding_matrix


def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect