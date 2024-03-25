import numpy as np
import time
from torch_geometric.loader import DataLoader
import argparse
import math

from octis.models.GraphNTM.modules import GDGNNModel
from octis.models.GraphNTM.dataPrepare.graph_data import GraphDataset
from octis.models.GraphNTM.dataPrepare.preprocess import process_dataset, clean_vocab, select_embedding
from octis.models.GraphNTM.utils import eval_top_doctopic, eval_topic, print_top_pairwords, save_edge

import torch
from octis.models.model import AbstractModel
import os
import pandas as pd
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class GNTM(AbstractModel):

    def __init__(
        self, dataset='', model_type='GDGNNMODEL', prior_type='Dir2', enc_nh=128,
        num_topic=20, batch_size=100, optimizer='Adam', learning_rate=0.001, momentum=0.9,
        num_epoch=10, init_mult=1.0, eval=False, taskid=0, load_path='', ni=300, nw=300,
        fixing=True, STOPWORD=False, nwindow=5, prior=1.0, num_samp=1, MIN_TEMP=0.3,
        INITIAL_TEMP=1.0, maskrate=0.5, wdecay=1e-4, word=True, variance=0.995,
        seed=2023, use_partitions=True, use_validation=False,
        save_dir=None, save_path=None, log_path=None, data_path=None):

        super().__init__()

        self.hyperparameters['dataset'] = dataset
        self.hyperparameters['model_type'] = model_type
        self.hyperparameters['prior_type'] = prior_type
        self.hyperparameters['enc_nh'] = enc_nh
        self.hyperparameters['num_topic'] = num_topic
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['optimizer'] = optimizer
        self.hyperparameters['learning_rate'] = learning_rate
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['num_epoch'] = num_epoch
        self.hyperparameters["init_mult"] = init_mult
        self.hyperparameters["eval"] = eval
        self.hyperparameters["taskid"] = taskid
        self.hyperparameters["load_path"] = load_path
        self.hyperparameters['ni'] = ni
        self.hyperparameters["nw"] = nw
        self.hyperparameters["fixing"] = fixing
        self.hyperparameters["STOPWORD"] = STOPWORD
        self.hyperparameters["nwindow"] = nwindow
        self.hyperparameters["prior"] = prior
        self.hyperparameters['num_samp'] = num_samp
        self.hyperparameters["MIN_TEMP"] = MIN_TEMP
        self.hyperparameters["INITIAL_TEMP"] = INITIAL_TEMP
        self.hyperparameters["maskrate"] = maskrate
        self.hyperparameters["wdecay"] = wdecay
        self.hyperparameters["word"] = word
        self.hyperparameters["variance"] = variance

        self.hyperparameters['seed'] = seed

        self.hyperparameters['save_dir'] = save_dir
        self.hyperparameters['save_path'] = save_path
        self.hyperparameters['log_path'] = log_path

        self.data_path = data_path+'/'+dataset+'/'

        self.use_partitions = use_partitions
        self.use_validation = use_validation

        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        np.random.seed(self.hyperparameters['seed'])
        torch.manual_seed(self.hyperparameters['seed'])
        random.seed(self.hyperparameters['seed'])

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.hyperparameters['seed'])
            torch.cuda.manual_seed_all(self.hyperparameters['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def run_test(self, model, test_loader, mode='VAL', verbose=False):
        model.eval()  # switch to testing mode
        num_sent = 0
        val_output = {}
        for batch in test_loader:
            batch = batch.to(self.device)
            batch_size = batch.y.size(0)
            outputs = model.loss(batch)
            for key in outputs:
                if key not in val_output:
                    val_output[key] = 0
                val_output[key] += outputs[key].item() * batch_size
            num_sent += batch_size

        if verbose:
            report_str = ' ,'.join(['{} {:.4f}'.format(key, val_output[key] / num_sent) for key in val_output])
            print('--{} {} '.format(mode, report_str))
        return val_output['loss'] / num_sent
    
    def learn_feature(self, model, loader):
        model.eval()  # switch to testing mode
        thetas = []
        labels = []
        for batch in loader:
            batch = batch.to(self.device)
            theta = model.get_doctopic(batch)
            thetas.append(theta)
            labels.append(batch.y)
        thetas = torch.cat(thetas, dim=0).detach()
        labels = torch.cat(labels, dim=0).detach()
        return thetas,labels
    
    def eval_doctopic(self, model, test_loader):
        thetas, labels = self.learn_feature(model,test_loader)
        thetas=thetas.cpu().numpy()
        labels = labels.cpu().numpy()
        eval_top_doctopic(thetas, labels)

    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        trains GNTM model

        :param dataset: octis Dataset for training the model
        :param hyperparameters: dict, with optionally) the following information:
        :param top_words: number of top-n words of the topics (default 10)

        """

        clip_grad = 20.0
        decay_epoch = 5
        lr_decay = 0.8
        max_decay = 5
        ANNEAL_RATE =0.00003

        if hyperparameters is None:
            hyperparameters = {}
        
        self.set_params(hyperparameters)
        # self.vocab = dataset.get_vocabulary()
        args = GNTM.dict2args(self.hyperparameters)
        args = GNTM.get_paths(args)
        args.device = self.device
        
        if self.use_partitions and self.use_validation:
            train, validation, test = dataset.get_partitioned_corpus(use_validation=True)

            tr_text = [' '.join(i) for i in train]
            val_text = [' '.join(i) for i in validation]
            ts_text = [' '.join(i) for i in test]

            all_labels = dataset.get_labels()

            label2id = {w: i for i, w in enumerate(set(all_labels))}

            tr_labels = [label2id[label] for label in all_labels[:len(train)]]
            va_labels = [label2id[label] for label in all_labels[len(train):len(train)+len(validation)]]
            ts_labels = [label2id[label] for label in all_labels[-len(test):]]


            train_data, val_data, test_data,\
                args.vocab, whole_edge, \
                whole_edge_w, stop_str = self.preprocess(train=tr_text, test=ts_text, validation=val_text,
                                                         dataname=args.dataset,
                                                         tr_l=tr_labels, va_l=va_labels, ts_l=ts_labels,
                                                         save_path=self.data_path)
            
            args.vocab_size = len(args.vocab)
            print('edge number: %d' % whole_edge.shape[0])
            save_edge(whole_edge.cpu().numpy(), whole_edge_w, args.vocab.id2word_, args.save_dir + '/whole_edge.csv')
            
            word_vec = np.load(self.data_path + '{}d_words{}.npy'.format(args.nw, stop_str))
            word_vec = torch.from_numpy(word_vec).float()

            train_loader = DataLoader(train_data, batch_size=self.hyperparameters['batch_size'], shuffle=False, follow_batch=['x', 'edge_id', 'y'], worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_data, batch_size=self.hyperparameters['batch_size'], shuffle=False, follow_batch=['x', 'edge_id', 'y'], worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_data, batch_size=self.hyperparameters['batch_size'], shuffle=False, follow_batch=['x', 'edge_id', 'y'], worker_init_fn=seed_worker, generator=g)

            model = GDGNNModel(args, word_vec=word_vec, whole_edge=whole_edge).to(self.device)   

            print('paramteres', sum(param.numel() for param in model.parameters()))
            print('trainable paramteres', sum(param.numel() for param in model.parameters() if param.requires_grad==True))
            
            if args.eval:
                args.temp = args.MIN_TEMP
                print('begin evaluation')
                if args.load_path != '':
                    model.load_state_dict(torch.load(args.load_path, map_location=torch.device(self.device)))
                    print("%s loaded" % args.load_path)
                else:
                    model.load_state_dict(torch.load(args.save_path, map_location=torch.device(self.device)))
                    print("%s loaded" % args.save_path)
                model.eval()
                self.run_test(model, test_loader, 'TEST')
                torch.cuda.empty_cache()

                beta = model.get_beta().detach().cpu().numpy()
                data = pd.read_csv(self.data_path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                # data = data[data['train']>0]
                common_texts = [text for text in data['content'].values]
                eval_topic(beta, [args.vocab.id2word(i) for i in range(args.vocab_size)], common_texts= dataset.get_corpus())

            ALTER_TRAIN = True
            # model.set_embedding(word_vec, fix=False)
            if args.optimizer == 'Adam':
                enc_optimizer = torch.optim.Adam(model.enc_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                weight_decay=args.wdecay)
                dec_optimizer = torch.optim.Adam(model.dec_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                weight_decay=args.wdecay)
            else:
                assert False, 'Unknown optimizer {}'.format(args.optimizer)
            
            best_loss = 1e4
            iter_ = decay_cnt = 0
            args.iter_ = iter_
            args.temp = args.INITIAL_TEMP
            opt_dict = {"not_improved": 0, "lr": args.learning_rate, "best_loss": 1e4}
            log_niter = len(train_loader) // 5
            start = time.time()
            args.iter_threahold = max(30*len(train_loader), 2000)
            
            for epoch in range(args.num_epoch):
                num_sents = 0
                output_epoch = {}
                model.train()  # switch to training mode
                for batch in train_loader:
                    batch = batch.to(self.device)
                    batch_size = batch.y.size(0)
                    outputs = model.loss(batch)
                    loss = outputs['loss']
                    num_sents += batch_size
                    # optimize

                    dec_optimizer.zero_grad()
                    enc_optimizer.zero_grad()

                    loss.backward()  # backprop
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    if ALTER_TRAIN:
                        if epoch % 2 == 0:
                            dec_optimizer.step()

                        else:
                            enc_optimizer.step()
                    else:
                        enc_optimizer.step()
                        dec_optimizer.step()
                    # report
                    for key in outputs:
                        if key not in output_epoch:
                            output_epoch[key] = 0
                        output_epoch[key] += outputs[key].item() * batch_size

                    # if iter_ % log_niter == 0:
                        # report_str = ' ,'.join(['{} {:.4f}'.format(key, output_epoch[key] / num_sents) for key in output_epoch])
                        # print('Epoch {}, iter {}, {}, time elapsed {:.2f}s'.format(epoch, iter_, report_str, time.time() - start))
                    iter_ += 1
                    args.iter_ = iter_
                    ntither=args.iter_-args.iter_threahold
                
                    if  ntither>=0 and ntither % 1000 == 0 and args.temp > args.MIN_TEMP:
                        args.temp = max(args.temp * math.exp(-ANNEAL_RATE * ntither), args.MIN_TEMP)
                        # args.temp = max(args.temp * ANNEAL_RATE, args.MIN_TEMP)
                        best_loss = 1e4
                        opt_dict["best_loss"] = best_loss
                        opt_dict["not_improved"] = 0
                        model.load_state_dict(torch.load(args.save_path))
                if ALTER_TRAIN and epoch%2==0:
                    continue

                model.eval()  # switch to testing mode
                with torch.no_grad():
                    val_loss = self.run_test(model, val_loader, 'VAL')
                print(best_loss,opt_dict["best_loss"], args.temp, ALTER_TRAIN)
                if val_loss < best_loss:
                    print('update best loss')
                    best_loss = val_loss
                    torch.save(model.state_dict(), args.save_path)
                    
                if val_loss > opt_dict["best_loss"]:
                    opt_dict["not_improved"] += 1
                    if opt_dict["not_improved"] >= decay_epoch and epoch >= 15 and args.temp==args.MIN_TEMP:
                        opt_dict["best_loss"] = best_loss
                        opt_dict["not_improved"] = 0
                        opt_dict["lr"] = opt_dict["lr"] * lr_decay
                        model.load_state_dict(torch.load(args.save_path))
                        print('new lr: %f' % opt_dict["lr"])
                        decay_cnt += 1
                        if args.optimizer == 'Adam':
                            enc_optimizer = torch.optim.Adam(model.enc_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                            weight_decay=args.wdecay)
                            dec_optimizer = torch.optim.Adam(model.dec_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                            weight_decay=args.wdecay)
                        else:
                            assert False, 'Unknown optimizer {}'.format(args.optimizer)
                else:
                    opt_dict["not_improved"] = 0
                    opt_dict["best_loss"] = val_loss
                if decay_cnt == max_decay:
                    break
                with torch.no_grad():
                    self.run_test(model, test_loader, 'TEST')
                    if epoch % 5 == 0:
                        # torch.cuda.empty_cache()
                        beta = model.get_beta().detach().cpu().numpy()
                        if epoch>0 and (epoch)%50==0:
                            data = pd.read_csv(self.data_path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                            # data = data[data['train']>0]
                            common_texts = [text for text in data['content'].values]
                            eval_topic(beta, [args.vocab.id2word(i) for i in range(args.vocab_size)], common_texts=common_texts)
                            if args.model_type in ['GDGNNMODEL']:
                                beta_edge = model.get_beta_edge(False).detach().cpu().numpy()[:, 1:]  # 第0位为非边的权重
                                # print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=args.vocab.id2word_)
                                for k in range(len(beta_edge)):
                                    save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=args.vocab.id2word_,
                                            fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
                                beta_edge = model.get_beta_edge(True).detach().cpu().numpy()[:, 1:]
                                # print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=args.vocab.id2word_)
                                for k in range(len(beta_edge)):
                                    save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=args.vocab.id2word_,
                                            fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
                            if args.model_type in ['GDGNNMODEL5']:
                                W = model.get_W().detach().cpu().numpy()
                                # print('W', W)
                        else:
                            data = pd.read_csv(self.data_path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                            # data = data[data['train']>0]
                            common_texts = [text for text in data['content'].values]
                            eval_topic(beta, [args.vocab.id2word(i) for i in range(args.vocab_size)], common_texts=common_texts)


                        if dataset.get_labels() is not None:
                            self.eval_doctopic(model, test_loader)
                        
                model.train()
            
            model.load_state_dict(torch.load(args.save_path))
            model.eval()
            with torch.no_grad():
                self.run_test(model, test_loader, 'TEST')
                torch.cuda.empty_cache()
                beta = model.get_beta().detach().cpu().numpy()
                data = pd.read_csv(self.data_path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                # data = data[data['train']>0]
                common_texts = [text for text in data['content'].values]
                eval_topic(beta, [args.vocab.id2word(i) for i in range(args.vocab_size)],  common_texts=common_texts)
                if dataset.get_labels() is not None:
                    self.eval_doctopic(model, test_loader)
                
                if args.model_type in [ 'GDGNNMODEL']:
                    beta_edge = model.get_beta_edge(False).detach().cpu().numpy()[:,1:] # 第0位为非边的权重
                    # print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=args.vocab.id2word_)
                    for k in range(len(beta_edge)):
                        save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=args.vocab.id2word_,
                                fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
                    beta_edge = model.get_beta_edge(True).detach().cpu().numpy()[:,1:]
                    # print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=args.vocab.id2word_)
                    for k in range(len(beta_edge)):
                        save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=args.vocab.id2word_,
                                fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
                if args.model_type in ['GDGNNMODEL']:
                    W = model.get_W().detach().cpu().numpy()
                    # print('W', W)

            tr_theta = self.get_thetas(model, train_loader).T
            ts_theta = self.get_thetas(model, test_loader).T

            result = self.get_info(args, beta, tr_theta, top_words)
            result['test-topic-document-matrix'] = ts_theta
            return result

        # elif self.use_partitions and not self.use_validation:
        #     train, test = dataset.get_partitioned_corpus(use_validation=False)
        #     x_train, x_test = self.preprocess(vocab=self.vocab,
        #                             mini_doc=self.hyperparameters["mini_doc"],
        #                             train=train,
        #                             test=test,
        #                             validation=None,
        #                             biterm_tr_path=self.graph_path + "window_length{}_train.pkl".format(self.hyperparameters['window_length']),
        #                             biterm_ts_path=self.graph_path + "window_length{}_test.pkl".format(self.hyperparameters['window_length']),
        #                             window_length=self.hyperparameters["window_length"])

        #     self._train_epoch(x_train)
        #     result = self.get_info(x_train, top_words)
        #     result['test-topic-document-matrix'] = np.asarray(self.get_thetas(x_test)).T
        #     return result

        # else:
        #     x_train = self.preprocess(vocab=self.vocab,
        #                     mini_doc=self.hyperparameters["mini_doc"],
        #                     train=train,
        #                     test=test,
        #                     validation=None,
        #                     biterm_tr_path=self.graph_path + "window_length{}_train.pkl".format(self.hyperparameters['window_length']),
        #                     window_length=self.hyperparameters["window_length"])

        #     self._train_epoch(x_train)
        #     result = self.get_info(x_train, top_words)
        #     return result

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results
    
    def get_info(self, args, beta, theta, top_words=10):
        info = {}
        info['topic-word-matrix'] = beta
        info['topic-document-matrix'] = theta
        info['topics'] = self.get_topics(beta, args.vocab.id2word_, n_top_words=top_words)
        return info
    
    def get_thetas(self, model, loader):
        model.eval()
        thetas = []
        for batch in loader:
            batch = batch.to(self.device)
            theta = model.get_doctopic(batch)
            thetas.append(theta)
        
        thetas = torch.cat(thetas, dim=0).detach()
        thetas = thetas.cpu().numpy()
        return thetas
    
    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions

    @staticmethod
    def dict2args(dict):
        parser = argparse.ArgumentParser(description='Arguments of GraphNTM')
        parser.add_argument("-f", required=False)
        for k, v in dict.items():
            parser.add_argument('--' + k, default=v)
        args = parser.parse_args()
        print(args)
        return args
    
    @staticmethod
    def get_paths(args):
        save_dir = args.save_dir + "/models/%s/%s_%s/" % (args.dataset, args.dataset, args.model_type)
        opt_str = '_%s_m%.2f_lr%.4f' % (args.optimizer, args.momentum, args.learning_rate)
        model_str = '_%s_ns%d_ench%d_ni%d_nw%d_ngram%d_temp%.2f-%.2f' % \
                    (args.model_type, args.num_samp, args.enc_nh, args.ni, args.nw,args.nwindow, args.INITIAL_TEMP,args.MIN_TEMP)
        
        if args.model_type in [ 'GDGNNMODEL']:
            id_ = '%s_topic%d%s_prior_type%s_%.2f%s_%d_%d_stop%s_fix%s_word%s' % \
                (args.dataset, args.num_topic, model_str, args.prior_type,args.prior,
                opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing), str(args.word))
        else:
            id_ = '%s_topic%d%s%s_%d_%d_stop%s_fix%s' % \
                (args.dataset, args.num_topic, model_str,
                opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing))

        save_dir += id_
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        args.save_dir = save_dir
        print("save dir", args.save_dir)
        args.save_path = os.path.join(save_dir, 'model.pt')
        args.log_path = os.path.join(save_dir, "log.txt")
        return args
    
    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
    
    @staticmethod
    def get_topics(beta, vocab, n_top_words=10):
        topic_w = []
        for k in range(beta.shape[0]):
            if np.isnan(beta[k]).any():
                topic_w = None
                break
            else:
                top_words = list(beta[k].argsort()[-n_top_words:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topic_w.append(topic_words)
        return topic_w

    @staticmethod
    def preprocess(train, test=None, validation=None, dataname=None,
                   tr_l=None, va_l=None, ts_l=None, save_path=None):

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        data, vocab = process_dataset(dataset=dataname, STOPWORD=False,
                                      tr=train, va=validation, ts=test,
                                      tr_l=tr_l, va_l=va_l, ts_l=ts_l)
        
        data.to_csv(save_path+'/overall.csv', header=True, index=False, quoting=1)
        clean_vocab(save_path, freq_threshold=5, STOPWORD=False)
        select_embedding(save_path, STOPWORD=False)

        dataset = GraphDataset(root=save_path, STOPWORD=False, ngram=5, edge_threshold=10)

        train_idxs = [i for i in range(len(dataset)) if dataset[i].train == 1]
        train_data = dataset[train_idxs]
        val_idxs = [i for i in range(len(dataset)) if dataset[i].train == -1]
        val_data = dataset[val_idxs]
        test_idxs = [i for i in range(len(dataset)) if dataset[i].train == 0]
        test_data = dataset[test_idxs]

        whole_edge = torch.tensor(dataset.whole_edge, dtype=torch.long, device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

        return train_data, val_data, test_data, dataset.vocab, \
            whole_edge, dataset.whole_edge_w, dataset.stop_str