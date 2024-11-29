from torch.utils.data import Dataset
import scipy.sparse
import torch

class GINOPICDataset(Dataset):

    def __init__(self, X_bow, X_graphs, idx2token):

        if X_bow.shape[0] != len(X_graphs):
            raise Exception("Wait! BoW and Document Graphs have different sizes! ")

        self.X_bow = X_bow
        self.X_graphs = X_graphs
        self.idx2token = idx2token

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X_bow[i]) == scipy.sparse.csr.csr_matrix:
            X_bow = torch.FloatTensor(self.X_bow[i].todense())
        else:
            X_bow = torch.FloatTensor(self.X_bow[i])
        
        graph = self.X_graphs[i]
        return_dict = {'X_bow': X_bow, 'graph': graph}

        return return_dict

    def __len__(self):
        """Return length of dataset."""
        return self.X_bow.shape[0]
