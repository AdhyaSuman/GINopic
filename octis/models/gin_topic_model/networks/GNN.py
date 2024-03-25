import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 out_feats,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, weight=True))
        
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, weight=True))
            
        # output layer
        self.layers.append(GraphConv(n_hidden, out_feats, weight=True))
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, edge_w=None):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_w)
        return h
    
    
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads, feat_drop=0.2, attn_drop=0.2)
        self.conv2 = GATConv(h_feats*num_heads, out_feats, 1, feat_drop=0.2, attn_drop=0.2)

    def forward(self, g, in_feat, edge_w=None):
        h = self.conv1(g, in_feat, edge_w)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)
        h = self.conv2(g, h, edge_w)
        h = h.squeeze() 
        return h


# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, aggregator_type='lstm'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type)
        self.conv2 = SAGEConv(h_feats, out_feats, aggregator_type)

    def forward(self, g, in_feat,  edge_w=None):
        h = self.conv1(g, in_feat,  edge_w)
        h = F.relu(h)
        h = self.conv2(g, h, edge_w)
        return h