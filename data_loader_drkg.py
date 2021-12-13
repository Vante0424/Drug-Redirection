import torch
from torch.utils.data import Dataset
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.strategy = self.p.strategy
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)
        # self.entities_for_1_n = torch.arange(self.p.num_ent, dtype=torch.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = ele['triple'], ele['label']

        if self.strategy == 'one_to_n':
            y = torch.zeros([self.p.num_ent], dtype=torch.float32) + self.p.label_smoothing
            for e2 in label:
                y[e2] = 1.0
            return triple[0], triple[1], 0, y

        elif self.strategy == 'one_to_x':
            neg_ent = self.get_neg_ent(triple, label)

            y = torch.zeros((neg_ent.shape[0]), dtype=torch.float32) + self.p.label_smoothing
            y[0] = 1.
            return triple[0], triple[1], neg_ent, y
        else:
            raise

    def get_neg_ent(self, triple, label):
        if self.strategy == 'one_to_x':
            pos_obj = triple[2]
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0

            neg_ent = np.random.choice(self.entities[mask], self.p.neg_num+1, replace=False)
            neg_ent[0] = pos_obj

        else:
            raise
        return neg_ent


class TestDataset(Dataset):
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = ele['triple'], ele['label']
        label = self.get_label(label)
        return triple[0], triple[1], triple[2], label

    def get_label(self, label):
        y = torch.zeros([self.p.num_ent], dtype=torch.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


if __name__ == '__main__':
    import time

