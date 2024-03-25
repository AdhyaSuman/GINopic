import torch
from octis.models.BAT_model.dataset import DocDataset
from multiprocessing import cpu_count
from octis.models.model import AbstractModel
import torch
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from octis.models.BAT_model.models.GSM import GSM as GaussianSM


class GSM(AbstractModel):

    def __init__(self, num_topics=10, num_epochs=100, batch_size=512, use_partitions=False, use_validation=False, num_samples=10):

        assert not(use_validation and use_partitions), "Validation data is not needed for GMNTM. Please set 'use_validation=False'."
        
        super(GSM, self).__init__()
        self.hyperparameters = dict()
        self.hyperparameters['num_topics'] = int(num_topics)
        self.hyperparameters['num_epochs'] = int(num_epochs)
        self.hyperparameters['batch_size'] = int(batch_size)
        self.early_stopping = None
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples

    def train_model(self, dataset, hyperparameters=None, top_words=10):

        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.top_word = top_words
        self.vocab = dataset.get_vocabulary()

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            x_train, x_test, input_size = self.preprocess(train, test)

            self.model = GaussianSM(bow_dim=input_size,
                             n_topic=self.hyperparameters['num_topics'],
                             device=self.device,
                             taskname=None)
            
            self.model.train(train_data=x_train,
                             batch_size=self.hyperparameters['batch_size'],
                             test_data=None,
                             num_epochs=self.hyperparameters['num_epochs'],
                             log_every=1e9)
            
            result = self.get_info()
            result['test-topic-document-matrix'] = self.model.get_doc_topic_distribution(x_test,
                                                                                         n_samples=self.num_samples,
                                                                                         batch_size=self.hyperparameters['batch_size']).T
        else:
            train = dataset.get_corpus()
            x_train, input_size = self.preprocess(train)
            
            self.model = GaussianSM(bow_dim=input_size,
                              n_topic=self.hyperparameters['num_topics'],
                              device=self.device,
                              taskname=None)

            self.model.train(train_data=x_train,
                             batch_size=self.hyperparameters['batch_size'],
                             test_data=None,
                             num_epochs=self.hyperparameters['num_epochs'],
                             log_every=1e9)

            result = self.get_info()
        
        result['topic-document-matrix'] = self.model.get_doc_topic_distribution(x_train,
                                                                                n_samples=self.num_samples,
                                                                                batch_size=self.hyperparameters['batch_size']).T
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def get_info(self):
        info = {}
        with torch.no_grad():
            idxes = torch.eye(self.hyperparameters['num_topics']).to(self.device)
            info['topic-word-matrix'] = self.model.get_topic_word_dist(normalize=False)
        info['topics'] = self.model.show_topic_words(topK=self.top_word)
        return info


    def set_default_hyperparameters(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    
    @staticmethod
    def preprocess(train, test=None, validation=None, use_tfidf=False):

        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        dictionary = Dictionary(entire_dataset)
        train_vec = [dictionary.doc2bow(doc) for doc in train]

        vocabsize = len(dictionary)
        train_data = DocDataset(train_vec, train, dictionary)

        if test is not None and validation is not None:
            test_vec = [dictionary.doc2bow(doc) for doc in test]
            test_data = DocDataset(test_vec, test, dictionary)

            valid_vec = [dictionary.doc2bow(doc) for doc in validation]
            valid_data = DocDataset(valid_vec, validation, dictionary)
            return train_data, test_data, valid_data, vocabsize
        
        if test is None and validation is not None:
            valid_vec = [dictionary.doc2bow(doc) for doc in validation]
            valid_data = DocDataset(valid_vec, validation, dictionary)
            return train_data, valid_data, vocabsize
        
        if test is not None and validation is None:
            test_vec = [dictionary.doc2bow(doc) for doc in test]
            test_data = DocDataset(test_vec, test, dictionary)
            return train_data, test_data, vocabsize
        
        if test is None and validation is None:
            return train_data, vocabsize