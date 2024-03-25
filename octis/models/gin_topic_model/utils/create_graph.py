import dgl
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sim

class SimGraph:
    def __init__(self, token2id: dict, emb_mat, epsilon=0.3):
        
        assert len(token2id)==emb_mat.shape[0], \
            "length of token2id should be == emb_mat.shape[0]"
        
        self.token2id = token2id
        self.id2token = {idx:w for w,idx in token2id.items()}
        self.emb_mat = emb_mat
        self.sim = sim(emb_mat)
        self.sim[self.sim < epsilon] = 0.0 #replacing all values < epsilon with zeros

    def create_graph(self, ids):
        graph_rep = {}
        adj = self.sim[np.ix_(ids,ids)]#.copy()
        graph_rep['src'], graph_rep['dst'] = np.nonzero(adj)
        graph_rep['weight'] = adj[np.nonzero(adj)]
        graph_rep['node_id'] = ids
#         graph_rep['H'] = self.emb_mat[ids]
        return graph_rep
        
    def _text_to_graph(self, text):
        local_ids = [self.token2id[word] for word in set(text.split())]
        graph_rep = self.create_graph(local_ids)
        dgl_graph = dgl.graph((graph_rep['src'], graph_rep['dst']), num_nodes=len(graph_rep['node_id']))
        dgl_graph.ndata['id'] = torch.from_numpy(np.array(graph_rep['node_id'], dtype=np.int64))
        dgl_graph.edata['weight'] = torch.from_numpy(np.array(graph_rep['weight'], dtype=np.float32))
#         dgl_graph.ndata['H'] = torch.from_numpy(np.array(graph_rep['H'], dtype=np.float32))
        return dgl_graph