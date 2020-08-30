# GOLD-NAS: Gradual, One-Level, Differentiable

This is an unofficial implementation of GOLD-NAS (https://arxiv.org/abs/2007.03331).

**This code is based on the implementation of  [DARTS](https://github.com/quark0/darts).**

## Requirements
- python 3
- pytorch >= 1.1.0
- torchvision

## Results
Please refer to the original paper for complete results.

### Usage
#### Search on CIFAR10

```
python ./cifar_search/train_search.py \\
```

**Note: in case that you do not have a GPU with 32GB memory, you can reduce the base channel number of search from 36 to 16. We tried and succeeded, but achieved slightly lower accuracy.**

#### Search on ImageNet

We did not implement ImageNet search due to limited computational resource. You can refer to PC-DARTS and embed the ImageNet search code into GOLD-NAS, which is not too difficult.

#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on CIFAR10:

```
python ./cifar_train/train.py \\
```
