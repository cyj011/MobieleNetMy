""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import timm
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from models import *
from safetensors.torch import load_file

from util.samplers import RASampler
from util import utils as utils
from util.optimizer import SophiaG, MARS
from util.engine import train_one_epoch, evaluate
from util.losses import DistillationLoss

from datasets import build_dataset
from datasets.threeaugment import new_data_aug_generator

from estimate_model import Predictor, Plot_ROC, OptAUC


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()

        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):

        identity = x

        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w

        return out

MOBILENETV4_PRETRAINED_ALIASES = {
    "mobilenetv4_conv_small_050": "mobilenetv4_conv_small_050.e3000_r224_in1k",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "mobilenetv4_conv_medium": "mobilenetv4_conv_medium.e500_r256_in1k",
    "mobilenetv4_conv_large": "mobilenetv4_conv_large.e600_r384_in1k",
    "mobilenetv4_conv_aa_large": "mobilenetv4_conv_aa_large.e600_r384_in1k",
    "mobilenetv4_conv_blur_medium": "mobilenetv4_conv_blur_medium.e500_r224_in1k",
    "mobilenetv4_hybrid_medium": "mobilenetv4_hybrid_medium.e500_r224_in1k",
    "mobilenetv4_hybrid_large": "mobilenetv4_hybrid_large.e600_r384_in1k",
}


class PartialDepthwiseConv2d(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            padding_mode="zeros",
            n_div=4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.n_div = n_div
        self.dim_conv = max(1, channels // n_div)
        self.dim_untouched = channels - self.dim_conv

        self.partial_conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=self.dim_conv,
            bias=bias,
            padding_mode=padding_mode,
        )

    @classmethod
    def from_conv(cls, conv, n_div=4):
        pconv = cls(
            channels=conv.in_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            n_div=n_div,
        )
        with torch.no_grad():
            pconv.partial_conv.weight.copy_(conv.weight[:pconv.dim_conv])
            if conv.bias is not None:
                pconv.partial_conv.bias.copy_(conv.bias[:pconv.dim_conv])
        return pconv

    def forward(self, x):
        x_conv, x_untouched = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x_conv = self.partial_conv(x_conv)
        return torch.cat((x_conv, x_untouched), dim=1)


def _replace_high_channel_depthwise_with_pconv(module, min_channels=512, n_div=4):
    replaced = 0
    for child_name, child in module.named_children():
        if child_name in {"dw_start", "dw_mid", "dw_end"} and hasattr(child, "conv"):
            conv = child.conv
            is_depthwise = (
                isinstance(conv, nn.Conv2d)
                and conv.in_channels == conv.out_channels
                and conv.groups == conv.in_channels
                and conv.kernel_size != (1, 1)
            )
            is_high_channel = isinstance(conv, nn.Conv2d) and conv.in_channels >= min_channels
            is_stride_one = isinstance(conv, nn.Conv2d) and conv.stride == (1, 1)
            if is_depthwise and is_high_channel and is_stride_one:
                child.conv = PartialDepthwiseConv2d.from_conv(conv, n_div=n_div)
                replaced += 1
                continue
        replaced += _replace_high_channel_depthwise_with_pconv(child, min_channels=min_channels, n_div=n_div)
    return replaced


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_prefix_if_present(state_dict, prefix):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_local_pretrained_backbone(backbone, pretrained_path):
    if not pretrained_path:
        return None
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"Local pretrained weight not found: {pretrained_path}")

    if pretrained_path.endswith(".safetensors"):
        checkpoint = load_file(pretrained_path, device="cpu")
    else:
        checkpoint = torch.load(pretrained_path, map_location="cpu")

    state_dict = _unwrap_state_dict(checkpoint)
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    state_dict = _strip_prefix_if_present(state_dict, "backbone.")

    for key in list(state_dict.keys()):
        if key.startswith("classifier.") or key.startswith("head."):
            del state_dict[key]

    return backbone.load_state_dict(state_dict, strict=False)


def disable_unused_backbone_head(backbone):
    for name in ("global_pool", "conv_head", "norm_head", "act2", "flatten", "classifier"):
        if hasattr(backbone, name):
            setattr(backbone, name, nn.Identity())


def _make_block_activation(activation_name, inplace=True):
    name = activation_name.lower()
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "hardswish":
        return nn.Hardswish(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported block activation: {activation_name}")


def _is_replaceable_activation(module):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU)):
        return True
    class_name = module.__class__.__name__.lower()
    return class_name in {"relu", "relu6", "hardswish", "hard_swish", "silu", "swish", "gelu"}


def _replace_block_activations(module, activation_name):
    if activation_name.lower() in {"", "none", "original"}:
        return 0

    replaced = 0
    for child_name, child in module.named_children():
        if _is_replaceable_activation(child):
            setattr(module, child_name, _make_block_activation(activation_name))
            replaced += 1
        else:
            replaced += _replace_block_activations(child, activation_name)
    return replaced


class MobileNetV4_CA_PConv(nn.Module):

    def __init__(
            self,
            num_classes,
            model_name="mobilenetv4_conv_large",
            pretrained=True,
            pretrained_path="",
            pconv=True,
            pconv_n_div=4,
            pconv_min_channels=512,
            block_act="original"):

        super().__init__()
        use_online_pretrained = pretrained and not pretrained_path
        backbone_name = MOBILENETV4_PRETRAINED_ALIASES.get(model_name, model_name) if use_online_pretrained else model_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=use_online_pretrained,
            num_classes=0
        )

        self.local_pretrained_msg = load_local_pretrained_backbone(self.backbone, pretrained_path)
        disable_unused_backbone_head(self.backbone)

        self.pconv_stats = {
            "replaced_depthwise": 0,
            "n_div": pconv_n_div,
            "min_channels": pconv_min_channels,
            "local_pretrained": pretrained_path or "",
        }
        if pconv:
            self.pconv_stats["replaced_depthwise"] = _replace_high_channel_depthwise_with_pconv(
                self.backbone,
                min_channels=pconv_min_channels,
                n_div=pconv_n_div,
            )

        self.block_act_stats = {
            "activation": block_act,
            "replaced": _replace_block_activations(self.backbone.blocks, block_act) if hasattr(self.backbone, "blocks") else 0,
            "scope": "backbone.blocks",
        }

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise RuntimeError("Cannot infer MobileNetV4 feature dimension from backbone.num_features")

        self.ca = CoordAtt(feature_dim)

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

    def forward(self, x):

        x = self.backbone.forward_features(x)

        x = self.ca(x)

        x = x.mean((2, 3))

        x = self.classifier(x)

        return x

def get_args_parser():
    parser = argparse.ArgumentParser(
        'MobileNetV4 training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--predict', default=False, type=bool, help='plot ROC curve and confusion matrix')
    parser.add_argument('--opt_auc', default=False, type=bool, help='Optimize AUC')

    # Model parameters
    parser.add_argument('--model', default='mobilenetv4_conv_large', type=str, metavar='MODEL',
                        choices=['mobilenetv4_hybrid_large', 'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large_075',
                                'mobilenetv4_conv_large', 'mobilenetv4_conv_aa_large', 'mobilenetv4_conv_medium',
                                 'mobilenetv4_conv_aa_medium', 'mobilenetv4_conv_small', 'mobilenetv4_hybrid_medium_075',
                                 'mobilenetv4_conv_small_035', 'mobilenetv4_conv_small_050', 'mobilenetv4_conv_blur_medium'],
                        help='Name of model to train')
    parser.add_argument('--extra_attention_block', default=False, type=bool, help='Add an extra attention block')
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='Use ImageNet pretrained MobileNetV4 weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Disable online ImageNet pretrained weights')
    parser.add_argument('--pretrained_path', default='', type=str,
                        help='Local pretrained weight path, such as /path/to/model.safetensors')
    parser.add_argument('--pconv', default=True, type=str2bool,
                        help='Replace later high-channel depthwise convolutions with PConv')
    parser.add_argument('--pconv_n_div', default=4, type=int,
                        help='Only 1 / pconv_n_div channels use spatial convolution')
    parser.add_argument('--pconv_min_channels', default=512, type=int,
                        help='Only replace depthwise convolutions whose channel count is at least this value')
    parser.add_argument('--block_act', default='original',
                        choices=['original', 'relu6', 'hardswish', 'silu', 'gelu'],
                        help='Replace non-identity activations inside MobileNetV4 backbone blocks')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--adamw_lr', type=float, default=3e-3, metavar='AdamWLR',
                        help='Using MARS optimizer, learning rate for adamw(default: 3e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str,
                        default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha',
                        default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--freeze_layers', type=bool, default=False, help='freeze layers')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # Dataset parameters
    parser.add_argument('--data_root', default=r'D:\WorkPlace\pythonProgram\Emotion\fer2013\fer2013_img', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number classes of your dataset')
    parser.add_argument('--data-set', default='myData', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output_dir', default=r'D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\fe2013\mn+ca+pconv',
                        help='path where to save, empty for no saving')
    parser.add_argument('--writer_output', default=r'D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\fe2013\mn+ca+pconv',
                        help='path where to save SummaryWriter, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')


    return parser




def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError(
            "Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    dataset_train, dataset_val = build_dataset(args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,

        persistent_workers=False,
        prefetch_factor=4,
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,

        persistent_workers=False,
        prefetch_factor=4,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")


    # model = create_model(
    #     args.model,
    #     extra_attention_block=args.extra_attention_block,
    #     args=args
    # )
    import timm


    # ========================
    #  ✅ 修复：关闭预训练 + 正确插入 LCA 注意力
    # ========================
    model = MobileNetV4_CA_PConv(
        num_classes=args.nb_classes,
        model_name=args.model,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        pconv=args.pconv,
        pconv_n_div=args.pconv_n_div,
        pconv_min_channels=args.pconv_min_channels,
        block_act=args.block_act,
    )
    print(f"PConv changes: {model.pconv_stats}")
    print(f"Block activation changes: {model.block_act_stats}")
    if model.local_pretrained_msg is not None:
        print(f"Loaded local pretrained weights: {model.local_pretrained_msg}")



    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint
        # state_dict = model.state_dict()
        # new_state_dict = utils.map_safetensors(checkpoint_model, state_dict)

        for k in list(checkpoint_model.keys()):
            if 'classifier' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'classifier' not in name:
                    para.requires_grad_(False)
                # else:
                #     print('training {}'.format(name))
            if args.extra_attention_block:
                for name, para in model.extra_attention_block.named_parameters():
                    para.requires_grad_(True)
            if args.pconv:
                for module in model.modules():
                    if isinstance(module, PartialDepthwiseConv2d):
                        for para in module.parameters():
                            para.requires_grad_(True)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    #
    # print('*****************')
    # print('Initial LR is ', linear_scaled_lr)
    # print('*****************')

    # optimizer = create_optimizer(args, model_without_ddp)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=2e-4, weight_decay=args.weight_decay) if args.finetune else create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is
    # 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    max_accuracy = 0.0

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    latest_ckpt = f'{args.output_dir}/{args.model}_latest_checkpoint.pth'

    if args.resume or os.path.exists(latest_ckpt):

        args.resume = latest_ckpt

        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume,
                map_location='cpu',
                check_hash=True
            )
        else:
            print(f"Loading local checkpoint at {args.resume}")

            checkpoint = torch.load(
                args.resume,
                map_location='cpu',
                weights_only=False
            )


        # 加载模型
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)  # 👈 加个 strict=False 防报错
        print(msg)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)  # 👈 把 .cuda() 改成 .to(device) 更稳定

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            max_accuracy = checkpoint['best_score']
            print(f'Now max accuracy is {max_accuracy}')
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 将从 Epoch {args.start_epoch} 继续训练")  # 👈 加个提示

            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

            print("✅✅✅ 训练已从断点恢复完成！")

    if args.eval:
        # util.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        print(f'No Visualization')
        test_stats = evaluate(data_loader_val, model, device, None, None, args, visualization=False)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
    # print(model)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            # set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval,  # set bn to eval if finetune
            writer=writer,
            args=args
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, epoch, writer, args, visualization=True)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir:
            latest_ckpt_path = os.path.join(
                output_dir,
                f'{args.model}_latest_checkpoint.pth'
            )

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_score': max_accuracy,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, latest_ckpt_path)

            print(f"✅ Latest checkpoint saved: {latest_ckpt_path}")

        # =========================
        # 保存 best checkpoint
        # =========================
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]

            best_ckpt_path = os.path.join(
                output_dir,
                f'{args.model}_best_checkpoint.pth'
            )

            print(f"🏆 Saving BEST checkpoint to {best_ckpt_path}")

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_score': max_accuracy,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, best_ckpt_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # plot ROC curve and confusion matrix
    if args.predict and utils.is_main_process():
        model_predict = MobileNetV4_CA_PConv(
            num_classes=args.nb_classes,
            model_name=args.model,
            pretrained=False,
            pretrained_path="",
            pconv=args.pconv,
            pconv_n_div=args.pconv_n_div,
            pconv_min_channels=args.pconv_min_channels,
            block_act=args.block_act,
        )
        model_predict.to(device)
        print('*******************STARTING PREDICT*******************')
        Predictor(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)
        Plot_ROC(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)

        if args.opt_auc:
            OptAUC(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MobileNetV4 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
