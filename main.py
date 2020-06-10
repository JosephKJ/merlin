import argparse
import time
import pprint
import random

from lib.config import cfg_from_file
from lib.utils import *
from lib.baseline_adpater import run_experiment
from lib.merlin import learn_continually

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')


def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)


def main():
    parser = argparse.ArgumentParser(description='Meta Consolidation for Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/splitMNIST.yml')

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    if not os.path.exists('output'):
        os.makedirs('output')

    timestamp = time.strftime("%m%d_%H%M%S")
    cfg.timestamp = timestamp

    output_dir = './output/' + cfg.run_label + '_' + cfg.timestamp
    cfg.output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/models')
        os.makedirs(output_dir + '/encoded_models')
        os.makedirs(output_dir + '/pickles')
        os.makedirs(output_dir + '/pickles/recall')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/plots')

    logging.basicConfig(filename=output_dir + '/logs/' + timestamp + '.log', level=logging.DEBUG,
                        format='%(levelname)s:\t%(message)s')

    log(pprint.pformat(cfg))

    gpu_list = cfg.gpu_ids.split(',')
    gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(gpus[0]))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.continual.method.run_merlin:
        learn_continually()

    if cfg.continual.method.run_single_model:
        run_experiment('single')

    if cfg.continual.method.run_ewc:
        run_experiment('ewc')

    if cfg.continual.method.run_gem:
        run_experiment('gem')

    if cfg.continual.method.run_icarl:
        run_experiment('icarl')

    if cfg.continual.method.run_gss:
        run_experiment('GSS_IQP_Rehearse')

    if cfg.continual.method.run_gss_greedy:
        run_experiment('GSS_Greedy')


if __name__ == '__main__':
    main()
