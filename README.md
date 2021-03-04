# Meta-Consolidation for Continual Learning 
### To appear at NeurIPS 2020

### arXiv link: https://arxiv.org/abs/2010.00352

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
@inproceedings{NEURIPS2020_a5585a4d,
 author = {K J, Joseph and N Balasubramanian, Vineeth},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {14374--14386},
 publisher = {Curran Associates, Inc.},
 title = {Meta-Consolidation for Continual Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/a5585a4d4b12277fee5cad0880611bc6-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
