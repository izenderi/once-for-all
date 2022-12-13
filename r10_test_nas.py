import argparse
import numpy as np
import os
import random
import time
import math
from matplotlib import pyplot as plt

import horovod.torch as hvd
import torch
import torch.nn as nn
from torchvision import transforms, datasets

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.run_manager import DistributedImageNetRunConfig, RunManager
from ofa.utils import MyRandomResizedCrop

CHECK_POINT_FILE_PATH = 'exp/kernel_depth2kernel_depth_width/phase2/checkpoint/checkpoint.pth.tar'
BEST_FILE_PATH = 'exp/kernel_depth2kernel_depth_width/phase2/checkpoint/model_best.pth.tar'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    type=str,
    default="depth",
    choices=[
        "kernel",
        "depth",
        "expand",
    ],
)
parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
parser.add_argument("--resume", action="store_true")


# args = parser.parse_args()
args, unknown = parser.parse_known_args() # Use this one for ipykernel to avoid compile error
if args.task == "kernel":
    args.path = "exp/normal2kernel"
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = "3,5,7"
    args.expand_list = "6"
    args.depth_list = "4"
elif args.task == "depth":
    args.path = "exp/kernel2kernel_depth/phase%d" % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "6"
        args.depth_list = "2,3,4"
elif args.task == "expand":
    args.path = "exp/kernel_depth2kernel_depth_width/phase%d" % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "4,6"
        args.depth_list = "2,3,4"
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = "3,5,7"
        args.expand_list = "3,4,6"
        args.depth_list = "2,3,4"
else:
    raise NotImplementedError
args.manual_seed = 0

args.lr_schedule_type = "cosine"

args.base_batch_size = 64
args.valid_size = 10000

args.opt_type = "sgd"
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = "bn#bias"
args.fp16_allreduce = False

args.model_init = "he_fout"
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = "tf"
args.image_size = "128,160,192,224"
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = "proxyless"

args.width_mult_list = "1.0"
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = "ce"

#---------------------------------------------------------------

# Initialize Horovod
hvd.init()
# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

num_gpus = hvd.size()

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)

# image size
args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
if len(args.image_size) == 1:
    args.image_size = args.image_size[0]
MyRandomResizedCrop.CONTINUOUS = args.continuous_size
MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

# build run config from args
args.lr_schedule_param = None
args.opt_param = {
    "momentum": args.momentum,
    "nesterov": not args.no_nesterov,
}
args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
if args.warmup_lr < 0:
    args.warmup_lr = args.base_lr
args.train_batch_size = args.base_batch_size
args.test_batch_size = args.base_batch_size * 4
run_config = DistributedImageNetRunConfig(
    **args.__dict__, num_replicas=num_gpus, rank=hvd.rank()
)

# build net from args
args.width_mult_list = [
    float(width_mult) for width_mult in args.width_mult_list.split(",")
]
args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
args.expand_list = [int(e) for e in args.expand_list.split(",")]
args.depth_list = [int(d) for d in args.depth_list.split(",")]

args.width_mult_list = (
    args.width_mult_list[0]
    if len(args.width_mult_list) == 1
    else args.width_mult_list
)
net = OFAMobileNetV3(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(args.bn_momentum, args.bn_eps),
    dropout_rate=args.dropout,
    base_stage_width=args.base_stage_width,
    width_mult=args.width_mult_list,
    ks_list=args.ks_list,
    expand_ratio_list=args.expand_list,
    depth_list=args.depth_list,
)

# checkpoint = torch.load(CHECK_POINT_FILE_PATH)
checkpoint = torch.load(BEST_FILE_PATH)

net.load_state_dict(state_dict=checkpoint['state_dict'])

# run_manager = RunManager(".tmp/eval_subnet", net, run_config, init=False)
# # assign image size: 128, 132, ..., 224
# run_config.data_provider.assign_active_img_size(224)
# run_manager.reset_running_statistics(net=net)

# print("Test random subnet:")
# print(net.module_str)

# loss, (top1, top5) = run_manager.validate(net=net)
# print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))


#-------------------------------------

from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

cuda_available = torch.cuda.is_available()

# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)

#---------------------------------------
""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
net.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(net.module_str)