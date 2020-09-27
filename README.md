# Meta-Consolidation for Continual Learning 
### (NeurIPS 2020)

The ability to continuously learn and adapt itself to new tasks, without losing grasp of already acquired knowledge is a hallmark of biological learning systems, which current deep learning systems fall short of. In this work, we present a novel methodology for continual learning called MERLIN: Meta-Consolidation for Continual Learning.
We assume that weights of a neural network, for solving task, come from a meta-distribution. This meta-distribution is learned and consolidated incrementally. We operate in the challenging online continual learning setting, where a data point is seen by the model only once.
Our experiments with continual learning benchmarks of MNIST, CIFAR-10, CIFAR-100 and Mini-ImageNet datasets show consistent improvement over five baselines, including a recent state-of-the-art, corroborating the promise of MERLIN.

We herein share the code to replicate experiments with Split MNIST dataset.

#### Usage

```python
python main.py
```

#### Requirements

```shell script
Python: 3.7.4
PyTorch: 1.4.0
TorchVision: 0.5.0
CUDA Version: 10.1 
```

#### Citation

``` 
@incollection{NIPS2020_8296,
title = {Meta-Consolidation for Continual Learning},
author = {K J, Joseph and Balasubramanian, Vineeth},
booktitle = {Advances in Neural Information Processing Systems 34},
year = {2020}
}
```
