from sklearn.feature_extraction.text import TfidfVectorizer

from octis.models.model import AbstractModel
from octis.models.gin_topic_model.datasets import dataset
from octis.models.gin_topic_model.models import ginopic
from octis.models.gin_topic_model.utils.data_preparation import dgl_graph_from_list, w2v_from_list
import os
import pickle as pkl
import torch
import numpy as np
import random


class GINOPIC(AbstractModel):

    def __init__(
        self, num_topics=10, model_type='prodLDA', activation='softplus',
        dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
        solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
        prior_variance=None, num_layers=2, num_neurons=100, seed=None,
        use_partitions=True, use_validation=False, num_samples=10,
        inference_type="CombinedGraph", w2v_path=None, num_gin_layers=2, g_feat_size=128,
        num_mlp_layers=1, gin_hidden_dim=100, gin_output_dim=128, eps_simGraph=0.1, graph_path=None,
        graph_construction='SimilarityGraph'):
        """
        initialization of GINOPIC

        :param num_topics : int, number of topic components, (default 10)
        :param model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param activation : string, 'softplus', 'relu', 'sigmoid', 'swish',
            'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu' (default 'softplus')
        :param num_layers : int, number of layers (default 2)
        :param dropout : float, dropout to use (default 0.2)
        :param learn_priors : bool, make priors a learnable parameter
            (default True)
        :param batch_size : int, size of batch to use for training (default 64)
        :param lr : float, learning rate to use for training (default 2e-3)
        :param momentum : float, momentum to use for training (default 0.99)
        :param solver : string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_epochs : int, number of epochs to train for, (default 100)
        :param num_samples: int, number of times theta needs to be sampled
            (default: 10)
        :param seed : int, the random seed. Not used if None (default None).
        :param use_partitions: bool, if true the model will be trained on the
            training set and evaluated on the test set (default: true)
        :param reduce_on_plateau : bool, reduce learning rate by 10x on
            plateau of 10 epochs (default False)
        :param inference_type: the type of the GINOPIC model. (default CombinedGraph)
        
        """

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['activation'] = activation
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['inference_type'] = inference_type
        self.hyperparameters['learn_priors'] = learn_priors
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['lr'] = lr
        self.hyperparameters['num_samples'] = num_samples
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['solver'] = solver
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['reduce_on_plateau'] = reduce_on_plateau
        self.hyperparameters["prior_mean"] = prior_mean
        self.hyperparameters["prior_variance"] = prior_variance
        self.hyperparameters["num_neurons"] = num_neurons
        self.hyperparameters["num_layers"] = num_layers
        
        self.hyperparameters["g_feat_size"] = g_feat_size
        self.hyperparameters["num_gin_layers"] = num_gin_layers
        self.hyperparameters["num_mlp_layers"] = num_mlp_layers
        self.hyperparameters["gin_hidden_dim"] = gin_hidden_dim
        self.hyperparameters["gin_output_dim"] = gin_output_dim
        self.hyperparameters['eps_simGraph'] = eps_simGraph
        self.hyperparameters["seed"] = seed
        
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.w2v_path = w2v_path
        self.graph_path = graph_path+'_{}/'.format(graph_construction)

        self.graph_construction = graph_construction

        hidden_sizes = tuple([num_neurons for _ in range(num_layers)])
        self.hyperparameters['hidden_sizes'] = tuple(hidden_sizes)

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None, top_words=10, **ablation):
        """
        trains GINOPIC model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset.get_vocabulary()
        self.set_seed(seed=self.hyperparameters['seed'])

        if not os.path.isdir(self.w2v_path):
            os.makedirs(self.w2v_path)

        if not os.path.isdir(self.graph_path):
            os.makedirs(self.graph_path)

        if self.use_partitions and self.use_validation:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            data_corpus_validation = [' '.join(i) for i in validation]

            x_train, x_test, x_valid, input_size = self.preprocess(
                self.vocab, data_corpus_train, test=data_corpus_test, validation=data_corpus_validation,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                emb_dim=self.hyperparameters['g_feat_size'],
                eps_simGraph=self.hyperparameters['eps_simGraph'],
                graph_train_path=self.graph_path + "eps{}_F{}_train.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                graph_test_path=self.graph_path + "eps{}_F{}_test.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                graph_val_path=self.graph_path + "eps{}_F{}_val.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                method=self.graph_construction
                )

            self.model = ginopic.GINOPIC(
                input_size=input_size,
                g_feat_size=self.hyperparameters['g_feat_size'],
                model_type=self.hyperparameters['model_type'],
                num_topics=self.hyperparameters['num_topics'],
                dropout=self.hyperparameters['dropout'],
                activation=self.hyperparameters['activation'],
                lr=self.hyperparameters['lr'],
                inference_type=self.hyperparameters['inference_type'],
                hidden_sizes=self.hyperparameters['hidden_sizes'],
                solver=self.hyperparameters['solver'],
                momentum=self.hyperparameters['momentum'],
                num_epochs=self.hyperparameters['num_epochs'],
                learn_priors=self.hyperparameters['learn_priors'],
                batch_size=self.hyperparameters['batch_size'],
                num_samples=self.hyperparameters['num_samples'],
                topic_prior_mean=self.hyperparameters["prior_mean"],
                reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                topic_prior_variance=self.hyperparameters["prior_variance"],
                top_words=top_words,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                num_gin_layers=self.hyperparameters['num_gin_layers'],
                num_mlp_layers=self.hyperparameters['num_mlp_layers'],
                gin_hidden_dim=self.hyperparameters['gin_hidden_dim'],
                gin_output_dim=self.hyperparameters['gin_output_dim'],
                **ablation)

            self.model.fit(x_train, x_valid, verbose=False)
            result = self.inference(x_test)
            return result

        elif self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            x_train, x_test, input_size = self.preprocess(
                self.vocab, data_corpus_train, test=data_corpus_test,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                emb_dim=self.hyperparameters['g_feat_size'],
                eps_simGraph=self.hyperparameters['eps_simGraph'],
                graph_train_path=self.graph_path + "eps{}_F{}_train.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                graph_test_path=self.graph_path + "eps{}_F{}_test.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                method=self.graph_construction
                )
            
            self.model = ginopic.GINOPIC(
                input_size=input_size,
                g_feat_size=self.hyperparameters['g_feat_size'],
                model_type='prodLDA',
                num_topics=self.hyperparameters['num_topics'],
                dropout=self.hyperparameters['dropout'],
                activation=self.hyperparameters['activation'],
                lr=self.hyperparameters['lr'],
                inference_type=self.hyperparameters['inference_type'],
                hidden_sizes=self.hyperparameters['hidden_sizes'],
                solver=self.hyperparameters['solver'],
                momentum=self.hyperparameters['momentum'],
                num_epochs=self.hyperparameters['num_epochs'],
                learn_priors=self.hyperparameters['learn_priors'],
                batch_size=self.hyperparameters['batch_size'],
                num_samples=self.hyperparameters['num_samples'],
                topic_prior_mean=self.hyperparameters["prior_mean"],
                reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                topic_prior_variance=self.hyperparameters["prior_variance"],
                top_words=top_words,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                num_gin_layers=self.hyperparameters['num_gin_layers'],
                num_mlp_layers=self.hyperparameters['num_mlp_layers'],
                gin_hidden_dim=self.hyperparameters['gin_hidden_dim'],
                gin_output_dim=self.hyperparameters['gin_output_dim'],
                **ablation)

            self.model.fit(x_train, None, verbose=True)
            result = self.inference(x_test)
            return result

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            x_train, input_size = self.preprocess(
                self.vocab, data_corpus,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                emb_dim=self.hyperparameters['g_feat_size'],
                eps_simGraph=self.hyperparameters['eps_simGraph'],
                graph_train_path=self.graph_path + "eps{}_F{}_train.pkl".format(self.hyperparameters['eps_simGraph'], self.hyperparameters['g_feat_size']),
                method=self.graph_construction
                )

            self.model = ginopic.GINOPIC(
                input_size=input_size,
                g_feat_size=self.hyperparameters['g_feat_size'],
                model_type='prodLDA',
                num_topics=self.hyperparameters['num_topics'],
                dropout=self.hyperparameters['dropout'],
                activation=self.hyperparameters['activation'],
                lr=self.hyperparameters['lr'],
                inference_type=self.hyperparameters['inference_type'],
                hidden_sizes=self.hyperparameters['hidden_sizes'],
                solver=self.hyperparameters['solver'],
                momentum=self.hyperparameters['momentum'],
                num_epochs=self.hyperparameters['num_epochs'],
                learn_priors=self.hyperparameters['learn_priors'],
                batch_size=self.hyperparameters['batch_size'],
                num_samples=self.hyperparameters['num_samples'],
                topic_prior_mean=self.hyperparameters["prior_mean"],
                reduce_on_plateau=self.hyperparameters['reduce_on_plateau'],
                topic_prior_variance=self.hyperparameters["prior_variance"],
                top_words=top_words,
                w2v_path=self.w2v_path+"{}_emb.npz".format(self.hyperparameters['g_feat_size']),
                num_gin_layers=self.hyperparameters['num_gin_layers'],
                num_mlp_layers=self.hyperparameters['num_mlp_layers'],
                gin_hidden_dim=self.hyperparameters['gin_hidden_dim'],
                gin_output_dim=self.hyperparameters['gin_output_dim'],
                **ablation)

            self.model.fit(x_train, None, verbose=True)
            result = self.model.get_info()
            return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys() and k != 'hidden_sizes':
                self.hyperparameters[k] = hyperparameters.get(
                    k, self.hyperparameters[k])

        self.hyperparameters['hidden_sizes'] = tuple(
            [self.hyperparameters["num_neurons"] for _ in range(
                self.hyperparameters["num_layers"])])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def set_seed(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def preprocess(vocab, train, test=None, validation=None,
                   eps_simGraph=None, w2v_path=None, emb_dim=None,
                   graph_train_path=None, graph_test_path=None,
                   graph_val_path=None, method=None):
        
        print("eps_simGraph=", eps_simGraph)

        token2id = {w: i for i, w in enumerate(vocab)}
        vec = TfidfVectorizer(
            vocabulary=token2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

        x_train = vec.transform(train)

        if w2v_path is not None:
            if os.path.exists(w2v_path):
                emb_mat = np.load(w2v_path)
            else:
                emb_mat = w2v_from_list(entire_dataset, vocab, save_path=w2v_path, dim=emb_dim)

        # print('emb_mat.shape=', emb_mat.shape)
        g_train = GINOPIC.load_graphs(graph_train_path,
                                   texts=train,
                                   token2id=token2id,
                                   emb_mat=emb_mat,
                                   eps_simGraph=eps_simGraph,
                                   method=method)
        train_data = dataset.GINOPICDataset(x_train, g_train, idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            g_test = GINOPIC.load_graphs(graph_test_path,
                                      texts=test,
                                      token2id=token2id,
                                      emb_mat=emb_mat,
                                      eps_simGraph=eps_simGraph,
                                      method=method)
            test_data = dataset.GINOPICDataset(x_test, g_test, idx2token)

            x_valid = vec.transform(validation)
            g_valid = GINOPIC.load_graphs(graph_val_path,
                                       texts=validation,
                                       token2id=token2id,
                                       emb_mat=emb_mat,
                                       eps_simGraph=eps_simGraph,
                                       method=method)
            valid_data = dataset.GINOPICDataset(x_valid, g_valid, idx2token)
            return train_data, test_data, valid_data, input_size
        
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            g_valid = GINOPIC.load_graphs(graph_val_path,
                                       texts=validation,
                                       token2id=token2id,
                                       emb_mat=emb_mat,
                                       eps_simGraph=eps_simGraph,
                                       method=method)
            valid_data = dataset.GINOPICDataset(x_valid, g_valid, idx2token)
            return train_data, valid_data, input_size
        
        if test is not None and validation is None:
            x_test = vec.transform(test)
            g_test = GINOPIC.load_graphs(graph_test_path,
                                      texts=test,
                                      token2id=token2id,
                                      emb_mat=emb_mat,
                                      eps_simGraph=eps_simGraph,
                                      method=method)

            test_data = dataset.GINOPICDataset(x_test, g_test, idx2token)
            return train_data, test_data, input_size
        
        if test is None and validation is None:
            return train_data, input_size

    @staticmethod
    def load_graphs(graph_path, texts, token2id, method='DependencyParser',
                    emb_mat=None, eps_simGraph=None):
        if graph_path is not None:
            if os.path.exists(graph_path):
                graphs = pkl.load(open(graph_path, 'rb'))
            else:
                graphs = dgl_graph_from_list(texts, token2id, method,
                                             emb_mat=emb_mat, eps_simGraph=eps_simGraph)
                pkl.dump(graphs, open(graph_path, 'wb'))
        else:
            graphs = dgl_graph_from_list(texts, token2id, method,
                                         emb_mat=emb_mat, eps_simGraph=eps_simGraph)
        return graphs