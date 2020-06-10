import torch.utils.data as data
from lib.utils import log
from lib.config import cfg

import torch
import os


class Kernels(data.Dataset):
    def __init__(self, root, task='None', only_fc=False):

        self.root = root
        self.only_fc = only_fc
        self.task = task

        self.data = []

        path = os.path.join(self.root, '../pickles/dataset.pkl')
        if os.path.exists(path) and not cfg.kernels.rebuild_dataset_pickle:
            log('Loading preprocessed kernels from %s' % path)
            self.data = torch.load(path)
        else:
            self.data = self.load_data()
        log('Number of weights loaded: %d' % len(self.data))

    def __getitem__(self, index):
        weights = self.data[index]
        return weights

    def __len__(self):
        return len(self.data)

    def get_weight_len(self):
        return len(self.data[0])

    def add_a_weight(self, weight):
        vector = []
        for x in weight:
            vector.append(x)

        self.data.append(vector)
        log('Added one more weight to dataset. Current number of weights: %d' % len(self.data))

    def load_data(self):
        data = []
        for idx, model_name in enumerate(os.listdir(self.root)):
            model_path = os.path.join(self.root, model_name)
            if 'task_'+str(self.task) in model_path:
                log('\nExtracting weights from %s' % model_path)

                model_weights = torch.load(model_path)['model_state_dict']
                vec = self.kernels_to_vector(model_weights)
                data.extend([vec])

        path = os.path.join(self.root, '../pickles/dataset.pkl')
        torch.save(data, path)

        return data

    def kernels_to_vector(self, model_weights):
        vector = []
        for layer, weight in model_weights.items():
            size = int(torch.prod(torch.tensor(weight.size())).item())
            vector.extend(weight.view(size))
        return vector
