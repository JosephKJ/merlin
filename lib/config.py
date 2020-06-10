import numpy as np
from easydict import EasyDict as edict

root = edict()
cfg = root

root.run_label = 'merlin'
root.gpu_ids = '0'
root.verbose = False
root.device = None

root.seed = 180

root.model = 'ResNet32'

# Training Dynamics
root.epochs = 2
root.within_batch_update_count = 5
root.n_models = 2
root.learning_rate = 0.001
root.momentum = 0.9
root.weight_decay = 0.0001
root.batch_size_train = 512
root.batch_size_test = 100
root.input_size = 0
root.is_cifar_10 = False
root.is_cifar_100 = False
root.is_mini_imagenet = False

# Placeholders
root.timestamp = 'placeholder'
root.output_dir = 'placeholder'

# Kernels
root.kernels = edict()
root.kernels.dataset_path = ''
root.kernels.batch_size = 32
root.kernels.epochs = 5
root.kernels.learning_rate = 0.001
root.kernels.lr_decay_step = 1100000000
root.kernels.gamma = 0.5
root.kernels.threshold = 0.3
root.kernels.rebuild_dataset_pickle = False
root.kernels.point_estimate = False
root.kernels.encoder = 'ae'
root.kernels.intermediate_test = False
root.kernels.latent_dimension = 2
root.kernels.n_pseudo_weights = 1
root.kernels.n_finetune_src_models = 1
root.kernels.ensembling = edict()
root.kernels.ensembling.min_clf_accuracy = 80
root.kernels.ensembling.max_num_of_models = 50
root.kernels.use_classification_loss = False
root.kernels.use_reconstruction_loss = True
root.kernels.chunking = edict()
root.kernels.chunking.enable = False
root.kernels.chunking.chunk_size = 200
root.kernels.chunking.hidden_size = 100
root.kernels.chunking.last_chunk_size = 0
root.kernels.chunking.num_chunks = 0

# Recall
root.recall = edict()
root.recall.base_model = ''
root.recall.model_path = ''
root.recall.batch_size = 32
root.recall.use_generated_weights = True
root.recall.cumulative_prior = True

root.continual = edict()
root.continual.task = 'permuted_mnist'
root.continual.shuffle_task = False
root.continual.shuffle_datapoints = False
root.continual.rebuild_dataset = False
root.continual.n_tasks = 10
root.continual.n_class_per_task = 0
root.continual.samples_per_task = 1000
root.continual.validation_samples_per_task = 1000
root.continual.epochs = 10
root.continual.n_finetune_epochs = 10
root.continual.finetune_learning_rate = 0.001
root.continual.learning_rate = 0.001
root.continual.batch_size_train = 128
root.continual.batch_size_test = 128

root.continual.method = edict()
root.continual.method.run_merlin = True
root.continual.method.run_ewc = False
root.continual.method.run_gem = False
root.continual.method.run_icarl = False
root.continual.method.run_single_model = False
root.continual.method.run_gss = False
root.continual.method.run_gss_greedy = False


def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, root)
