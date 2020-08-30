import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import *
import math
from utils import drop_path


def flops_computation(ci, c, op_id, skip_in_reduction):
    UNIT = 0.000001
    CH = 32
    KS = 3
    ratio = c / ci
    if op_id == 1:
        return UNIT * 2 * (ci * ci * CH * CH + KS * KS * CH * CH * ci / ratio)
    elif op_id == 0:
        if skip_in_reduction:
            return UNIT * ci * ci * CH * CH
        else:
            return 0
    else:
        return 0


def node_computation(weights_node, eta_min):
    weight_sum = weights_node.sum()
    ops = 0
    for edge in weights_node:
        for w_op in edge:
            if w_op / weight_sum > eta_min:
                ops += 1
    return weight_sum, ops


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
        self.stride = stride

    def forward(self, x, weights, drop_prob, eta_min, node_sum):
        mix_op = 0
        k = 0
        for w, op in zip(weights, self._ops):
            if w > eta_min * node_sum:
                if not isinstance(op, Identity):
                    mix_op = mix_op + w * drop_path(op(x), drop_prob)
                else:
                    mix_op = mix_op + w * op(x)
            else:
                if not isinstance(self._ops[k], Zero):
                    self._ops[k] = Zero(self.stride)
            k += 1
        return mix_op


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, drop_prob, eta_min):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        # print('s0', s0.shape, 's1', s1.shape)
        # print(self._ops)
        offset = 0
        for i in range(self._steps):
            W = weights[offset:(offset + len(states))]
            weight_sum = W.sum()
            # for s in states:
            #     print(type(s))
            #     if type(s) is int:
            #         print(s)
            # print()
            s = sum(self._ops[offset + j](h, weights[offset + j], drop_prob, eta_min, weight_sum) for j, h in
                    enumerate(states))
            offset += len(states)
            # print('s', type(s))
            states.append(s)
        # for i, s in enumerate(states):
        #     print(i, end=' ')
        #     print(s, end=' ')
        #     print(s.shape)
        # assert False
        return torch.cat(states[2:], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(nn.Dropout(0), nn.Linear(768, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, eta_min, reg_flops, mu, steps=4, multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self.reg_flops = reg_flops
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._auxiliary = auxiliary
        self.eta_min = eta_min
        self.mu = mu
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == layers * 2 // 3:
                C_aux = C_prev
        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_aux, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()

    # def forward(self, input):
    #     logits_aux = None
    #     C = self._C
    #     s0 = s1 = self.stem(input)
    #
    #     cost_network = 0
    #     for i, cell in enumerate(self.cells):
    #         weights = torch.sigmoid(self._arch_parameters[i])
    #         if i in [self._layers // 3, 2 * self._layers // 3]:
    #             C *= 2
    #             reduction = True
    #         else:
    #             reduction = False
    #         edge_id = 0
    #         reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]
    #         for w in weights:
    #             op_id = 0
    #             cost_edge = 0
    #             for w_op in w:
    #                 if edge_id in reduction_list and reduction:
    #                     skip_in_reduction = True
    #                 else:
    #                     skip_in_reduction = False
    #                 weight_sum = 0
    #                 ops = 0
    #                 if edge_id == 0:
    #                     weight_sum, ops = node_computation(weights[0:2], self.eta_min)
    #                 elif edge_id == 2:
    #                     weight_sum, ops = node_computation(weights[2:5], self.eta_min)
    #                 elif edge_id == 5:
    #                     weight_sum, ops = node_computation(weights[5:9], self.eta_min)
    #                 elif edge_id == 9:
    #                     weight_sum, ops = node_computation(weights[9:14], self.eta_min)
    #                 if (w_op / weight_sum) > self.eta_min:
    #                     cost_edge += torch.log(1 + ops * w_op / weight_sum) * (
    #                             self.reg_flops + self.mu * flops_computation(self._C, C, op_id, skip_in_reduction))
    #                 op_id += 1
    #             cost_network += cost_edge
    #             edge_id += 1
    #         s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob, self.eta_min)
    #         if i == self._layers * 2 // 3:
    #             if self._auxiliary:
    #                 logits_aux = self.auxiliary_head(s1)
    #     out = self.global_pooling(s1)
    #     logits = self.classifier(out.view(out.size(0), -1))
    #     return logits, logits_aux, cost_network

    def forward(self, input):
        logits_aux = None
        flops = 0
        C = self._C
        s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            weights = F.sigmoid(self._arch_parameters[i])
            if i in [self._layers // 3, 2 * self._layers // 3]:
                C *= 2
                reduction = True
            else:
                reduction = False
            edge_id = 0
            reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]
            for w in weights:
                edge = 0
                op_id = 0
                for w_op in w:
                    if edge_id in reduction_list and reduction:
                        reduce_skip = True
                    else:
                        reduce_skip = False
                    if edge_id == 0:
                        nodes, ops = node_computation(weights[0:2], self.eta_min)
                    elif edge_id == 2:
                        nodes, ops = node_computation(weights[2:5], self.eta_min)
                    elif edge_id == 5:
                        nodes, ops = node_computation(weights[5:9], self.eta_min)
                    elif edge_id == 9:
                        nodes, ops = node_computation(weights[9:14], self.eta_min)
                    if (w_op / nodes) > self.eta_min:
                        edge += torch.log(1 + ops * w_op / nodes) * (
                                self.reg_flops + self.mu * flops_computation(self._C, C, op_id, reduce_skip))
                    op_id += 1
                flops += edge
                edge_id += 1
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob, self.eta_min)
            if i == 2 * self._layers // 3:
                if self._auxiliary:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux, flops

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        numix_ops = len(PRIMITIVES)
        self._arch_parameters = []
        for i in range(self._layers):
            self.alphas_temp = Variable(torch.zeros(k, numix_ops).cuda(), requires_grad=True)
            self._arch_parameters.append(self.alphas_temp)

    def current_flops(self):
        cost_network = 0
        C = self._C
        for i in range(self._layers):
            weights = F.sigmoid(self._arch_parameters[i])
            if i in [self._layers // 3, self._layers * 2 // 3]:
                C *= 2
                reduction = True
            else:
                reduction = False
            edge_id = 0
            reduction_list = [0, 1, 2, 3, 5, 6, 9, 10]
            for w in weights:
                op_id = 0
                cost_edge = 0
                for w_op in w:
                    if edge_id in reduction_list and reduction:
                        skip_in_reduction = True
                    else:
                        skip_in_reduction = False
                    weight_sum = 0
                    ops = 0
                    if edge_id == 0:
                        weight_sum, ops = node_computation(weights[0:2], self.eta_min)
                    elif edge_id == 2:
                        weight_sum, ops = node_computation(weights[2:5], self.eta_min)
                    elif edge_id == 5:
                        weight_sum, ops = node_computation(weights[5:9], self.eta_min)
                    elif edge_id == 9:
                        weight_sum, ops = node_computation(weights[9:14], self.eta_min)
                    if w_op / weight_sum > self.eta_min:
                        cost_edge += flops_computation(self._C, C, op_id, skip_in_reduction)
                    op_id += 1
                cost_network += cost_edge
                edge_id += 1
        return cost_network

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                weight_sum = W.sum()
                edge_id = 0
                for w_edge in W:
                    op_id = 0
                    for w_op in w_edge:
                        if w_op > self.eta_min * weight_sum:
                            gene.append((PRIMITIVES[op_id], edge_id, i + 2))
                        op_id += 1
                    edge_id += 1
                start = end
                n += 1
            return gene

        gene_list = []
        for i in range(self._layers):
            gene_list.append(_parse(F.sigmoid(self._arch_parameters[i]).data.cpu().numpy()))
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype._make([gene_list, concat])
        return genotype
