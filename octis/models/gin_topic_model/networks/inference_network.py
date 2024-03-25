from collections import OrderedDict
from torch import nn
import torch
from dgl.nn import SumPooling
from octis.models.gin_topic_model.networks.gin import GIN
from octis.models.gin_topic_model.networks.GNN import GCN, GAT, GraphSAGE

class InferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, g_feat_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2,
                 num_gin_layers=2, num_mlp_layers=2, gin_hidden_dim=100, gin_output_dim=768):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(InferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        print("-"*100)
        print("GNN Settings: \n\
               num_gin_layers: {}\n\
               num_mlp_layers: {}\n\
               input_dim: {}\n\
               hidden_dim: {}\n\
               output_dim: {}\n\
               learn_eps: {}\n\
               neighbor_pooling_type: {}".format(
                num_gin_layers, num_mlp_layers,
                g_feat_size, gin_hidden_dim,
                gin_output_dim, True, 'mean'
                ))
        print("-"*100)

        self.gnn = GIN(num_layers=num_gin_layers,
                       num_mlp_layers=num_mlp_layers,
                       input_dim=g_feat_size,
                       hidden_dim=gin_hidden_dim,
                       output_dim=gin_output_dim,
                       learn_eps=True,
                       neighbor_pooling_type='mean')
            
       
        self.nodepool=SumPooling()
        
        self.adapt_graph = nn.Linear(gin_output_dim, input_size)
        self.input_layer = nn.Linear(input_size + gin_output_dim, hidden_sizes[0])
        
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
      
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, graph, h):
        """Forward pass."""
        w = graph.edata['weight'] if 'weight' in graph.edata.keys() else None
        node_emb = self.gnn(graph, h, w)
        x_g = self.nodepool(graph, node_emb)
        # x_g = self.adapt_graph(x_g)
        x = torch.cat((x, x_g), 1)
        
        x = self.input_layer(x)
        #Encoder
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma

class GNNInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, g_feat_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2,
                 gnn_arch='GCN', num_gnn_layers = 2,
                 gnn_hidden_dim=100, gnn_output_dim=768):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(GNNInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        if gnn_arch == 'GCN':
            print("-"*100)
            print('\n\
                  gnn_arch: {}\n\
                  in_feats: {}\n\
                  n_hidden: {}\n\
                  n_layers: {}\n\
                  out_feats: {}\n\
                  activation: {},\n\
                  dropout={}'.format(gnn_arch,
                                     g_feat_size,
                                     gnn_hidden_dim,
                                     num_gnn_layers,
                                     gnn_output_dim,
                                     activation,
                                     dropout))
            print("-"*100)
            
            self.gnn = GCN(
                        in_feats=g_feat_size,
                        n_hidden=gnn_hidden_dim,
                        n_layers=num_gnn_layers,
                        out_feats=gnn_output_dim,
                        activation=self.activation,
                        dropout=dropout
                                  )
            
        elif gnn_arch == 'GAT':
            print("-"*100)
            print('\n\
                  gnn_arch: {}\n\
                  in_feats:{}\n\
                  h_feats:{}\n\
                  out_feats={}\n\
                  num_heads={}'.format(gnn_arch, g_feat_size, gnn_hidden_dim, gnn_output_dim, 3))
            print("-"*100)
            
            self.gnn = GAT(
                        in_feats=g_feat_size,
                        h_feats=gnn_hidden_dim,
                        out_feats=gnn_output_dim,
                        num_heads=3
                            )
        
        elif gnn_arch == 'GraphSAGE':
            print("-"*100)
            print('\n\
                  gnn_arch: {}\n\
                  in_feats: {}\n\
                  h_feats:{}\n\
                  out_feats={}\n\
                  aggregator_type={}'.format(gnn_arch,
                                            g_feat_size,
                                            gnn_hidden_dim,
                                            gnn_output_dim,
                                            'pool'
                                            ))
            print("-"*100)
            
            self.gnn = GraphSAGE(
                                in_feats=g_feat_size,
                                h_feats=gnn_hidden_dim,
                                out_feats=gnn_output_dim,
                                aggregator_type='pool'
                                  )
       
        self.nodepool=SumPooling()
        
        self.adapt_graph = nn.Linear(gnn_output_dim, input_size)
        self.input_layer = nn.Linear(input_size + input_size, hidden_sizes[0])
        
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
      
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, graph, h):
        """Forward pass."""
        w = graph.edata['weight'] if 'weight' in graph.edata.keys() else None
        node_emb = self.gnn(graph, h, w)
        x_g = self.nodepool(graph, node_emb)
        x_g = self.adapt_graph(x_g)
        x = torch.cat((x, x_g), 1)
        
        x = self.input_layer(x)
        #Encoder
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
