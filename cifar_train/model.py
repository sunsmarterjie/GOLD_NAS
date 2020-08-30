import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, concat, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    if reduction:
      op_names, indices_input, indices_output = zip(*genotype)
    else:
      op_names, indices_input, indices_output = zip(*genotype)
    self._compile(C, op_names, indices_input, indices_output, concat, reduction)

  def _compile(self, C, op_names, indices_input, indices_output, concat, reduction):
    assert len(op_names) == len(indices_input)
    self._steps = len(op_names) // 2
    self._concat = list(set(indices_output))
    self.multiplier = len(self._concat)
    self._ops = nn.ModuleList()
    for name, index_input, index_output in zip(op_names, indices_input, indices_output):
      stride = 2 if reduction and index_input < 2 else 1
      op = OPS[name](C, stride, False)
      self._ops += [op]
    self._indices_input = indices_input
    self._indices_output = indices_output

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    for i in range(4):
      s=0
      for j in range(len(self._indices_output)):
        if self._indices_output[j]==(i+2):
          h=states[self._indices_input[j]]
          op=self._ops[j]
          h=op(h)
          if self.training and drop_prob > 0.:
            if not isinstance(op, Identity):
              h = drop_path(h, drop_prob)
          s=s+h
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    genotype_arch = genotype.gene
    layers = len(genotype_arch)
    self._layers = len(genotype_arch)
    self._auxiliary = auxiliary
    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      genotype1 = genotype_arch[i]
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      concat=eval("genotype.%s" % "concat")
      cell = Cell(genotype1, concat, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev
    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux