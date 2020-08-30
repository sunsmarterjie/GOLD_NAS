import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
import random

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--searched_epochs', type=int, default=25, help='num of searched epochs')
parser.add_argument('--learning_rate_omega', type=float, default=0.01, help='learning rate for omega')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--learning_rate_alpha', type=float, default=1, help='learning rate for alpha')
parser.add_argument('--weight_decay_alpha', type=float, default=0, help='weight decay for alpha')
parser.add_argument('--eta_min', type=float, default=0.01, help='eta min')
parser.add_argument('--eta_max', type=float, default=0.05, help='eta max')
parser.add_argument('--pruning_n0', type=int, default=4, help='pruning velocity')
parser.add_argument('--lambda0', type=float, default=1e-5, help='lambda0')
parser.add_argument('--c0', type=float, default=2.0, help='c0')
parser.add_argument('--mu', type=float, default=0, help='the mu parameter')
parser.add_argument('--reg_flops', type=float, default=1, help='reg for FLOPs')
parser.add_argument('--min_flops', type=float, default=240, help='min FLOPs')
parser.add_argument('--base_flops', type=float, default=166.01862399999789, help='base FLOPs')
parser.add_argument('--auto_augment', action='store_false', default=True, help='whether autoaugment is used')
parser.add_argument('--stable_round', type=float, default=3, help='number of rounds for stability')

args, unparsed = parser.parse_known_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def prune_op(model, args):
    MAX = 10000.0
    EPS = 1e-8
    pruned_ops = 0
    for x in range(args.pruning_n0):
        pruning_cell = 0
        pruning_edge = 0
        pruning_op = 0
        pruning_w = MAX
        for cell_id in range(args.layers):
            cell_weights = torch.sigmoid(model.arch_parameters()[cell_id])
            edge_id = 0
            while edge_id < 14:
                edge_weights = cell_weights[edge_id]
                weight_sum = 0
                if edge_id == 0:
                    weight_sum = cell_weights[0:2].sum()
                elif edge_id == 2:
                    weight_sum = cell_weights[2:5].sum()
                elif edge_id == 5:
                    weight_sum = cell_weights[5:9].sum()
                elif edge_id == 9:
                    weight_sum = cell_weights[9:14].sum()
                op_id = 0
                for w_op in edge_weights:
                    w_normalized = w_op / weight_sum
                    if w_normalized > args.eta_min:
                        if w_normalized < pruning_w:
                            pruning_cell = cell_id
                            pruning_edge = edge_id
                            pruning_op = op_id
                            pruning_w = w_normalized
                    elif EPS < w_normalized <= args.eta_min:
                        pruned_ops += 1
                        logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e', cell_id, edge_id, op_id,
                                     w_normalized)
                        model._arch_parameters[cell_id].data[edge_id][op_id] -= MAX
                        weight_sum -= w_op
                    op_id += 1
                edge_id += 1
        if pruning_w > args.eta_max:
            pass
        else:
            pruned_ops += 1
            logging.info('Pruning (cell, edge, op) = (%d, %d, %d): at weight %e', pruning_cell, pruning_edge,
                         pruning_op, pruning_w)
            model._arch_parameters[pruning_cell].data[pruning_edge][pruning_op] -= MAX
    return pruned_ops


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, args.eta_min, args.reg_flops,
                    args.mu)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer_alpha = torch.optim.SGD(
        model.arch_parameters(),
        args.learning_rate_alpha,
        momentum=args.momentum,
        weight_decay=args.weight_decay_alpha)
    optimizer_omega = torch.optim.SGD(
        model.parameters(),
        args.learning_rate_omega,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    epoch = 0
    flops_lambda = 0
    flops_lambda_delta = args.lambda0
    finished = False
    t = 0
    while not finished:
        epoch_start = time.time()
        lr = args.learning_rate_omega
        model.drop_path_prob = 0
        logging.info('epoch %d lr %e flops_weight %e', epoch, lr, flops_lambda)
        train_acc, train_obj = train(train_queue, model, criterion, optimizer_alpha, optimizer_omega, flops_lambda)
        logging.info('train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)
        pruning_epoch = prune_op(model, args)
        current_flops = model.current_flops() + args.base_flops
        logging.info('current model flops %e', current_flops)
        if pruning_epoch >= args.pruning_n0:
            flops_lambda_delta = args.lambda0
            flops_lambda = flops_lambda / args.c0
        else:
            flops_lambda_delta = flops_lambda_delta * args.c0
            flops_lambda = flops_lambda + flops_lambda_delta
        if current_flops < args.min_flops:
            finished = True
        if pruning_epoch == 0:
            t = t + 1
        else:
            if t > args.stable_round:
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
            t = 0
        epoch += 1


def train(train_queue, model, criterion, optimizer_alpha, optimizer_omega, flops_lambda):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        # get a random minibatch from the search queue with replacement
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)
        optimizer_alpha.zero_grad()
        optimizer_omega.zero_grad()
        logits, logits_aux, loss_flops = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss += flops_lambda * loss_flops
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_alpha.step()
        optimizer_alpha.zero_grad()
        optimizer_omega.step()
        optimizer_omega.zero_grad()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)
        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)
        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
