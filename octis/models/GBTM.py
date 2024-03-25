import numpy as np
from torch.autograd import Variable
import torch.cuda
import _pickle as pickle
import os
import itertools

from octis.models.GraphBTM.pytorch_model import ProdLDA
from octis.models.GraphBTM.dataloader import bitermsDataset
from octis.models.GraphBTM.sparseMM import (SparseMM,
                                        sparse_ones,
                                        to_sparse,
                                        sparse_diag)

import torch
import random
from octis.models.model import AbstractModel

class GBTM(AbstractModel):

    def __init__(
        self, num_topics=50, en1_units=100, en2_units=100, batch_size=100,
        optimizer='Adam', lr=0.002, momentum=0.99, num_epochs=200,
        variance=.995, betavariance=0.04, seed=2020,
        window_length=30, mini_doc=3, graph_path=None,
        use_partitions=True, use_validation=False,
        ):

        super().__init__()

        self.hyperparameters['en1_units'] = en1_units
        self.hyperparameters['en2_units'] = en2_units
        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['optimizer'] = optimizer
        self.hyperparameters['lr'] = lr
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters["variance"] = variance
        self.hyperparameters["betavariance"] = betavariance
        self.hyperparameters["window_length"] = window_length
        self.hyperparameters["mini_doc"] = mini_doc

        self.hyperparameters["seed"] = seed
        self.graph_path = graph_path
        self.use_partitions = use_partitions
        self.use_validation = use_validation

        self.model = None
        self.vocab = None
        self.beta = None

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    
    def _make_optimizer(self):
        if self.hyperparameters['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adadelta(self.model.params,
                                                  lr=1,
                                                  rho=0.99)
            
        elif self.hyperparameters['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             self.hyperparameters['lr'],
                                             momentum=self.hyperparameters['momentum'])
        else:
            assert False, 'Unknown optimizer {}'.format(self.hyperparameters['optimizer'])

    def _train_epoch(self, dataloader):
        for epoch in range(self.hyperparameters['num_epochs']):
            loss_epoch = 0.0
            self.model.train() # switch to training mode
            #total_batch = int(n_samples_tr / batch_size)
            sparsity = False
            count = 0
            b_count = 0
            GCNsInputList = []
            target_input = []
            data_size = dataloader.length
            for biterm in dataloader:
                mm = SparseMM.apply
                biterm = torch.FloatTensor(biterm).float().cuda()
                target_input.append(Variable(biterm))

                sparse_biterms = to_sparse(biterm.float().cuda())
                ones = torch.cuda.FloatTensor(biterm.shape[0]).fill_(1).unsqueeze(-1)

                indices = to_sparse(mm(sparse_biterms, ones))._indices()
                values = torch.cuda.FloatTensor(indices.size()[1]).fill_(1)

                adj_mask = torch.cuda.sparse.FloatTensor(indices, values, (sparse_biterms.size()[0], 1))

                eye = sparse_ones(biterm.size()[0]).cuda()

                adj = (sparse_biterms + eye).coalesce()

                degree_matrix = mm(adj, ones)
                degree_matrix = torch.pow(degree_matrix, -0.5)
                degree_matrix = degree_matrix * adj_mask.to_dense()
                degrees = sparse_diag(degree_matrix.squeeze(-1)).coalesce()

                adj = mm(adj, degrees.to_dense())
                adj = mm(degrees, adj)
                indices = (sparse_biterms + eye).coalesce()._indices()
                values = adj[tuple(indices[i] for i in range(indices.shape[0]))]
                adj = torch.cuda.sparse.FloatTensor(indices, values, sparse_biterms.size())

                GCNsInputList.append((Variable(sparse_biterms),
                                        Variable(adj)) )
                b_count += 1
                if b_count % self.hyperparameters['batch_size'] != 0:
                    continue
                if b_count > data_size:
                    break

                _, loss = self.model(GCNsInputList, None, compute_loss=True, l1=False, target=target_input)

                # optimize
                self.optimizer.zero_grad() # clear previous gradients
                loss.backward() # backprop
                #torch.nn.utils.clip_grad_norm(model.params, 5)
                self.optimizer.step() # update parameters
                # report
                loss_epoch += loss.data # add loss to loss_epoch
                GCNsInputList = []
                target_input = []
                if count % 10 == 0:
                    print('Epoch {}, count {}, loss={}'.format(epoch, count, loss_epoch / (count+1)))
                count += 1

            self.beta = torch.nn.functional.softmax(self.model.decoder.weight, 0).data.cpu().numpy().T
            # self.print_top_words(self.beta, self.vocab, n_top_words=10)
            
    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains GBTM model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """
        if hyperparameters is None:
            hyperparameters = {}
        
        self.set_params(hyperparameters)

        for k,v in self.hyperparameters.items():
            print(k,':',v)
        
        if not os.path.isdir(self.graph_path):
            os.makedirs(self.graph_path)

        self.vocab = dataset.get_vocabulary()
        self.set_seed(seed=self.hyperparameters['seed'])

        self.model = ProdLDA(num_input=len(self.vocab),
                             num_topic=self.hyperparameters['num_topics'],
                             en1_units=self.hyperparameters['en1_units'],
                             en2_units=self.hyperparameters['en2_units'])
        
        self.model = self.model.to(self.device)

        self._make_optimizer()

        if self.use_partitions and self.use_validation:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            x_train, x_test, x_valid = self.preprocess(vocab=self.vocab,
                                            mini_doc=self.hyperparameters["mini_doc"],
                                            train=train,
                                            test=test,
                                            validation=validation,
                                            biterm_tr_path=self.graph_path + "window_length{}_train.pkl".format(self.hyperparameters['window_length']),
                                            biterm_ts_path=self.graph_path + "window_length{}_test.pkl".format(self.hyperparameters['window_length']),
                                            biterm_va_path=self.graph_path + "window_length{}_valid.pkl".format(self.hyperparameters['window_length']),
                                            window_length=self.hyperparameters["window_length"])

            self._train_epoch(x_train)
            result = self.get_info(x_train, top_words)
            result['test-topic-document-matrix'] = np.asarray(self.get_thetas(x_test)).T
            return result

        elif self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            x_train, x_test = self.preprocess(vocab=self.vocab,
                                    mini_doc=self.hyperparameters["mini_doc"],
                                    train=train,
                                    test=test,
                                    validation=None,
                                    biterm_tr_path=self.graph_path + "window_length{}_train.pkl".format(self.hyperparameters['window_length']),
                                    biterm_ts_path=self.graph_path + "window_length{}_test.pkl".format(self.hyperparameters['window_length']),
                                    window_length=self.hyperparameters["window_length"])

            self._train_epoch(x_train)
            result = self.get_info(x_train, top_words)
            result['test-topic-document-matrix'] = np.asarray(self.get_thetas(x_test)).T
            return result

        else:
            x_train = self.preprocess(vocab=self.vocab,
                            mini_doc=self.hyperparameters["mini_doc"],
                            train=train,
                            test=test,
                            validation=None,
                            biterm_tr_path=self.graph_path + "window_length{}_train.pkl".format(self.hyperparameters['window_length']),
                            window_length=self.hyperparameters["window_length"])

            self._train_epoch(x_train)
            result = self.get_info(x_train, top_words)
            return result

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results
    
    def get_info(self, x_train, top_words=10):
        info = {}
        info['topic-word-matrix'] = self.beta
        info['topic-document-matrix'] = np.asarray(self.get_thetas(x_train)).T
        info['topics'] = self.get_topics(self.beta, self.vocab, n_top_words=10)
        return info
    
    @staticmethod
    def get_topics(beta, vocab, n_top_words=10):
        topic_w = []
        for k in range(beta.shape[0]):
                if np.isnan(beta[k]).any():
                    # to deal with nan matrices
                    topic_w = None
                    break
                else:
                    top_words = list(beta[k].argsort()[-n_top_words:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topic_w.append(topic_words)
        return topic_w
    
    @staticmethod
    def print_top_words(beta, feature_names, n_top_words=10):
        print( '---------------Printing the Topics------------------')
        for i in range(len(beta)):
            print(" ".join([feature_names[j+1]
                for j in beta[i][1:].argsort()[:-n_top_words - 1:-1]]))
        print( '---------------End of Topics------------------')
    
    def get_thetas(self, dataloader):
        self.model.eval()
        final_thetas = []
        with torch.no_grad():
            collect_theta = []
            count = 0
            b_count = 0
            GCNsInputList = []

            data_size = dataloader.length
            for biterm in dataloader:
                mm = SparseMM.apply
                biterm = torch.FloatTensor(biterm).float().cuda()

                sparse_biterms = to_sparse(biterm.float().cuda())
                ones = torch.cuda.FloatTensor(biterm.shape[0]).fill_(1).unsqueeze(-1)

                indices = to_sparse(mm(sparse_biterms, ones))._indices()
                values = torch.cuda.FloatTensor(indices.size()[1]).fill_(1)

                adj_mask = torch.cuda.sparse.FloatTensor(indices, values, (sparse_biterms.size()[0], 1))

                eye = sparse_ones(biterm.size()[0]).cuda()

                adj = (sparse_biterms + eye).coalesce()

                degree_matrix = mm(adj, ones)
                degree_matrix = torch.pow(degree_matrix, -0.5)
                degree_matrix = degree_matrix * adj_mask.to_dense()
                degrees = sparse_diag(degree_matrix.squeeze(-1)).coalesce()

                adj = mm(adj, degrees.to_dense())
                adj = mm(degrees, adj)
                indices = (sparse_biterms + eye).coalesce()._indices()
                values = adj[tuple(indices[i] for i in range(indices.shape[0]))]
                adj = torch.cuda.sparse.FloatTensor(indices, values, sparse_biterms.size())

                GCNsInputList.append((Variable(sparse_biterms),
                                        Variable(adj)))
                b_count += 1
                if b_count % self.hyperparameters['batch_size'] != 0:
                    continue
                if b_count > data_size:
                    break

                self.model.zero_grad()
                _ = self.model(GCNsInputList, None, compute_loss=False)
                collect_theta.extend(self.model.p.cpu().numpy().tolist())

                count += 1
                GCNsInputList = []
            final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0)

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions
    
    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

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
    def preprocess(vocab, mini_doc, train, test=None, validation=None,
                   biterm_tr_path=None, biterm_ts_path=None, biterm_va_path=None,
                   window_length=30):

        vocab2id = {w:i for i,w in enumerate(vocab)}

        data_tr = np.array([np.array([vocab2id[w] for w in doc]) for doc in train], dtype=object)
        if os.path.isfile(biterm_tr_path):
            print('loading biterms for training')
            with open(biterm_tr_path, 'rb') as f:
                biterms_tr = pickle.load(f, encoding='bytes')
        else:
            print('generating biterm graphs for training')
            biterms_tr = np.array(GBTM.make_biterm(data_tr, window_length, biterm_tr_path))
        
        tr_data = bitermsDataset(np.array(biterms_tr), len(vocab), mini_doc, data=None)

        if test is not None:
            data_ts = np.array([np.array([vocab2id[w] for w in doc]) for doc in test], dtype=object)
            if os.path.isfile(biterm_ts_path):
                print('loading biterms for testing')
                with open(biterm_ts_path, 'rb') as f:
                    biterms_ts = pickle.load(f, encoding='bytes')
            else:
                print('generating biterm graphs for testing')
                biterms_ts = np.array(GBTM.make_biterm(data_ts, window_length, biterm_ts_path))
            ts_data = bitermsDataset(np.array(biterms_ts), len(vocab), mini_doc, data=None)

            if validation is not None:
                data_va = np.array([np.array([vocab2id[w] for w in doc]) for doc in validation], dtype=object)
                if os.path.isfile(biterm_va_path):
                    print('loading biterms for validation')
                    with open(biterm_va_path, 'rb') as f:
                        biterms_va = pickle.load(f, encoding='bytes')
                else:
                    print('generating biterm graphs for validation')
                    biterms_va = np.array(GBTM.make_biterm(data_va, window_length, biterm_va_path))
                va_data = bitermsDataset(np.array(biterms_va), len(vocab), mini_doc, data=None)
                return tr_data, ts_data, va_data
            else:
                return tr_data, ts_data
        
        else:
            if validation is not None:
                data_va = np.array([np.array([vocab2id[w] for w in doc]) for doc in validation], dtype=object)
                if os.path.isfile(biterm_va_path):
                    print('loading biterms for validation')
                    with open(biterm_va_path, 'rb') as f:
                        biterms_va = pickle.load(f, encoding='bytes')
                else:
                    print('generating biterm graphs for validation')
                    biterms_va = np.array(GBTM.make_biterm(data_va, window_length, biterm_va_path))
                va_data = bitermsDataset(np.array(biterms_va), len(vocab), mini_doc, data=None)
                return tr_data, va_data
            else:
                return tr_data

    @staticmethod
    def make_biterm(data, window_length, save_path=None):
        biterms = []
        for doc in data:
            if np.sum(doc) == 0:
                continue
            doc_len = len(doc)
            temp = {}
            if doc_len > window_length:
                for i in range(doc_len - window_length):
                    for biterm in itertools.combinations(doc[i:i+window_length], 2):
                        GBTM.put_dic(frozenset(biterm), temp)
            else:
                for biterm in itertools.combinations(doc, 2):
                    GBTM.put_dic(frozenset(biterm), temp)
            biterms.append(temp)
        print('No. of biterms = ', len(biterms))
        with open(save_path, 'wb') as f:
            pickle.dump(biterms, f)
    
    @staticmethod
    def put_dic(biterm, biterms):
        if biterm not in biterms:
            biterms[biterm] = 1
        else:
            biterms[biterm] += 1