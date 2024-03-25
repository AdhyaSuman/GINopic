"""PyTorch class for feed foward AVITM network."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from octis.models.gin_topic_model.networks.inference_network import InferenceNetwork, GNNInferenceNetwork


class DecoderNetwork(nn.Module):

    def __init__(self, input_size, g_feat_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,learn_priors=True,
                 w2v_path=None, num_gin_layers=2, num_mlp_layers=2, gin_hidden_dim=100, gin_output_dim=768,
                 **ablation):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        self.num_gin_layers = num_gin_layers
        self.num_mlp_layers = num_mlp_layers
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_output_dim = gin_output_dim

        if infnet =='CombinedGraph':
            self.inf_net = InferenceNetwork(input_size,
                                g_feat_size,
                                n_components,
                                hidden_sizes,
                                activation,
                                dropout=0.2,
                                num_gin_layers=num_gin_layers,
                                num_mlp_layers=num_mlp_layers,
                                gin_hidden_dim=gin_hidden_dim,
                                gin_output_dim=gin_output_dim)
        
        elif infnet =='Ablation':
            self.inf_net = GNNInferenceNetwork(input_size,
                                g_feat_size,
                                n_components,
                                hidden_sizes,
                                activation,
                                dropout=0.2,
                                gnn_arch=ablation['gnn_arch'],
                                num_gnn_layers=ablation['num_gnn_layers'],
                                gnn_hidden_dim=ablation['gnn_hidden_dim'],
                                gnn_output_dim=ablation['gnn_output_dim'])

        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        
        #word_vec
        if w2v_path is None:
            print('Node embeddings is initialized using "xavier_uniform_"')
            self.word_vec = torch.Tensor(input_size, g_feat_size)
            if torch.cuda.is_available():
                self.word_vec = self.word_vec.cuda()
            self.word_vec = nn.Parameter(self.word_vec)
            nn.init.xavier_uniform_(self.word_vec)
        else:
            #load the numpy WordEmb matrix
            print('Node embeddings is initialized using Word2Vec')
            self.word_vec = torch.from_numpy(np.float32(np.load(w2v_path)))
            if torch.cuda.is_available():
                self.word_vec = self.word_vec.cuda()
            self.word_vec = nn.Parameter(self.word_vec)
            
            
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, graph):
        """Forward pass."""
        # batch_size x n_components
        feat = self.word_vec[graph.ndata['id']]
        posterior_mu, posterior_log_sigma = self.inf_net(x, graph, feat)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        
        topic_doc = theta
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            # word_dist: batch_size x input_size
            topic_word = self.beta
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            topic_word = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word, topic_doc

    def get_theta(self, x, graph):
        with torch.no_grad():
            # batch_size x n_components
            feat = self.word_vec[graph.ndata['id']]
            posterior_mu, posterior_log_sigma = self.inf_net(x, graph, feat)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta