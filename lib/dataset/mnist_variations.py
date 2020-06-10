import os
import random
import numpy as np
from PIL import Image
from six.moves import urllib
from sklearn.model_selection import train_test_split

from lib.config import cfg
from lib.utils import log

import torch
import torch.utils.data as data
from torch.utils.model_zoo import tqdm
from torchvision import transforms


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


class MNIST(data.Dataset):
    def __init__(self, root, task, mode='Train', transform=None, target_transform=None, tasks_so_far=None):
        """
        :param mode: Can be 'Train', 'Test' or 'Val'
        """
        super(MNIST, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode  # training set or test set

        self.variation = cfg.continual.task

        # Getting the MNIST dataset ready
        path = os.path.join(self.root, self.variation)
        if not os.path.exists(path):
            self.load_data(path)
            log('Downloaded and pickled MNIST data.')

        # Performing variations to MNIST
        variation_data = None
        if cfg.continual.task == 'permuted_mnist':
            save_location = os.path.join(path, 'processed', 'permuted_mnist.pt')
            if not os.path.exists(save_location) or cfg.continual.rebuild_dataset:
                self.create_permuted_mnist(path, save_location)
                cfg.continual.rebuild_dataset = False
            variation_data = torch.load(save_location)
        elif cfg.continual.task == 'rotated_mnist':
            save_location = os.path.join(path, 'processed', 'rotated_mnist.pt')
            if not os.path.exists(save_location) or cfg.continual.rebuild_dataset:
                self.create_rotated_mnist(path, save_location)
                cfg.continual.rebuild_dataset = False
            variation_data = torch.load(save_location)
        elif cfg.continual.task == 'split_mnist':
            save_location = os.path.join(path, 'processed', 'split_mnist_2.pt')
            if not os.path.exists(save_location) or cfg.continual.rebuild_dataset:
                self.create_split_mnist(path, save_location)
                cfg.continual.rebuild_dataset = False
            variation_data = torch.load(save_location)

        # Getting data ready
        self.data = []
        self.targets = []

        if self.mode == 'Train':
            if cfg.continual.samples_per_task < 0:
                self.data = variation_data[0][task][0]
                self.targets = variation_data[0][task][1]
            else:   # Sub-sample
                n_items = variation_data[0][task][0].size(0)
                indices = torch.randperm(n_items)[:cfg.continual.samples_per_task]
                self.data = variation_data[0][task][0].index_select(0, indices)
                self.targets = variation_data[0][task][1].index_select(0, indices)
        elif self.mode == 'Test':
            self.data = variation_data[1][task][0]
            self.targets = variation_data[1][task][1]
        elif self.mode == 'Val':
            if task == -1 and tasks_so_far is not None:
                for t in tasks_so_far:
                    n_items = variation_data[2][t][0].size(0)
                    indices = torch.randperm(n_items)[:cfg.continual.validation_samples_per_task]
                    self.data.extend(variation_data[2][t][0].index_select(0, indices))
                    self.targets.extend(variation_data[2][t][1].index_select(0, indices))
            else:
                n_items = variation_data[2][task][0].size(0)
                indices = torch.randperm(n_items)[:cfg.continual.validation_samples_per_task]
                self.data = variation_data[2][task][0].index_select(0, indices)
                self.targets = variation_data[2][task][1].index_select(0, indices)
        else:
            raise Exception('Invalid mode passed: %s' % self.mode)

    def __getitem__(self, index):
        img = self.data[index]
        target = int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def load_data(self, path):

        # URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
        url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
        data_path = os.path.join(path, 'data')

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            file_path = os.path.join(data_path, 'mnist.npz')
            try:
                print('Downloading ' + url + ' to ' + file_path)
                urllib.request.urlretrieve(
                    url, file_path,
                    reporthook=gen_bar_updater()
                )
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + file_path)
                    urllib.request.urlretrieve(
                        url, file_path,
                        reporthook=gen_bar_updater()
                    )
                else:
                    raise e

        processed_path = os.path.join(path, 'processed')
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
            file_path = os.path.join(data_path, 'mnist.npz')

            f = np.load(file_path)
            x_te = torch.from_numpy(f['x_test'])
            y_te = torch.from_numpy(f['y_test']).long()

            fraction = cfg.continual.validation_samples_per_task / len(f['x_train'])
            _, x_val, _, y_val = train_test_split(f['x_train'], f['y_train'], test_size=fraction, stratify=f['y_train'])

            x_tr = torch.from_numpy(f['x_train'])
            y_tr = torch.from_numpy(f['y_train']).long()

            f.close()

            # x_tr = torch.from_numpy(x_tr)
            # y_tr = torch.from_numpy(y_tr).long()

            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val).long()

            torch.save((x_tr, y_tr), os.path.join(processed_path, 'mnist_train.pt'))
            torch.save((x_te, y_te), os.path.join(processed_path, 'mnist_test.pt'))
            torch.save((x_val, y_val), os.path.join(processed_path, 'mnist_val.pt'))

    def create_permuted_mnist(self, path, save_location):
        log('Creating %d permutations of MNIST.' % cfg.continual.n_tasks)

        mnist_train_path = os.path.join(path, 'processed', 'mnist_train.pt')
        mnist_test_path = os.path.join(path, 'processed', 'mnist_test.pt')
        mnist_val_path = os.path.join(path, 'processed', 'mnist_val.pt')

        x_tr, y_tr = torch.load(mnist_train_path)
        x_te, y_te = torch.load(mnist_test_path)
        x_val, y_val = torch.load(mnist_val_path)

        x_tr = x_tr.view(-1, 28*28)
        x_te = x_te.view(-1, 28*28)
        x_val = x_val.view(-1, 28*28)

        train_tasks = []
        test_tasks = []
        val_tasks = []

        for task in range(cfg.continual.n_tasks):
            perm = torch.randperm(28*28)
            train_tasks.append([x_tr.index_select(1, perm).view(-1, 28, 28), y_tr])
            test_tasks.append([x_te.index_select(1, perm).view(-1, 28, 28), y_te])
            val_tasks.append([x_val.index_select(1, perm).view(-1, 28, 28), y_val])

        torch.save([train_tasks, test_tasks, val_tasks], save_location)

    def rotate_dataset(self, d, rotation):
        result = torch.FloatTensor(d.size(0), 28, 28)
        transfrm = transforms.ToTensor()

        for i in range(d.size(0)):
            img = Image.fromarray(d[i].numpy(), mode='L')
            result[i] = transfrm(img.rotate(rotation))
        return result

    def create_rotated_mnist(self, path, save_location):
        log('Creating %d rotations of MNIST.' % cfg.continual.n_tasks)

        mnist_train_path = os.path.join(path, 'processed', 'mnist_train.pt')
        mnist_test_path = os.path.join(path, 'processed', 'mnist_test.pt')
        mnist_val_path = os.path.join(path, 'processed', 'mnist_val.pt')

        x_tr, y_tr = torch.load(mnist_train_path)
        x_te, y_te = torch.load(mnist_test_path)
        x_val, y_val = torch.load(mnist_val_path)

        train_tasks = []
        test_tasks = []
        val_tasks = []

        min_rotation = 0.
        max_rotation = 180.

        for task in range(cfg.continual.n_tasks):
            min_rot = 1.0 * task / cfg.continual.n_tasks * (max_rotation - min_rotation) + min_rotation
            max_rot = 1.0 * (task + 1) / cfg.continual.n_tasks * (max_rotation - min_rotation) + min_rotation
            rot = random.random() * (max_rot - min_rot) + min_rot

            log('Rotating by %s degrees' % str(rot))

            train_tasks.append([self.rotate_dataset(x_tr, rot), y_tr])
            test_tasks.append([self.rotate_dataset(x_te, rot), y_te])
            val_tasks.append([self.rotate_dataset(x_val, rot), y_val])

        torch.save([train_tasks, test_tasks, val_tasks], save_location)

    def create_split_mnist(self, path, save_location):
        log('Creating the splits for Split-MNIST.')

        mnist_train_path = os.path.join(path, 'processed', 'mnist_train.pt')
        mnist_test_path = os.path.join(path, 'processed', 'mnist_test.pt')
        mnist_val_path = os.path.join(path, 'processed', 'mnist_val.pt')

        x_tr, y_tr = torch.load(mnist_train_path)
        x_te, y_te = torch.load(mnist_test_path)
        x_val, y_val = torch.load(mnist_val_path)

        class_per_task = int(10 / cfg.continual.n_tasks)
        train_tasks = []
        test_tasks = []
        val_tasks = []

        for t in range(cfg.continual.n_tasks):
            c1 = t * class_per_task
            c2 = (t + 1) * class_per_task
            i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
            i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
            i_val = ((y_val >= c1) & (y_val < c2)).nonzero().view(-1)

            train_tasks.append([x_tr[i_tr].clone(), y_tr[i_tr].clone()])
            test_tasks.append([x_te[i_te].clone(), y_te[i_te].clone()])
            val_tasks.append([x_val[i_val].clone(), y_val[i_val].clone()])

        torch.save([train_tasks, test_tasks, val_tasks], save_location)
