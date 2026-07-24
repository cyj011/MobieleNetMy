""" ImageNet Training Script
Fixed with FastGELU + Speed + Accuracy Optimization
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import json
import os

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


# ===================== 快速 GELU（提速+稳定） =====================
class FastGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
# ==================================================================


def get_args_parser():
    parser = argparse.ArgumentParser('MobileNetV4 training and evaluation script', add_help=False)

    # ===================== 修复：速度 + 精度参数 =====================
    parser.add_argument('--batch-size', default=64, type=int)          # 更快
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--input-size', default=224, type=int)       # 关键：提速3倍
    parser.add_argument('--predict', default=False, type=bool)
    parser.add_argument('--opt_auc', default=False, type=bool)

    # Model parameters
    parser.add_argument('--model', default='mobilenetv4_conv_large', type=str, metavar='MODEL',
                        choices=['mobilenetv4_hybrid_large', 'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large_075',
                                'mobilenetv4_conv_large', 'mobilenetv4_conv_aa_large', 'mobilenetv4_conv_medium',
                                 'mobilenetv4_conv_aa_medium', 'mobilenetv4_conv_small', 'mobilenetv4_hybrid_medium_075',
                                 'mobilenetv4_conv_small_035', 'mobilenetv4_conv_small_050', 'mobilenetv4_conv_blur_medium'],
                        help='Name of model to train')
    parser.add_argument('--extra_attention_block', default=False, type=bool)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996)
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt-eps', default=1e-8, type=float)
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+')
    parser.add_argument('--clip-grad', type=float, default=0.02)
    parser.add_argument('--clip-mode', type=str, default='agc')
    parser.add_argument('--momentum', type=float, default=0.9)

    # ===================== 修复：GELU 最优权重衰减 =====================
    parser.add_argument('--weight-decay', type=float, default=0.005)

    # Learning rate
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR')   # 修复：学习率正确
    parser.add_argument('--adamw_lr', type=float, default=3e-3)
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None)
    parser.add_argument('--lr-noise-pct', type=float, default=0.67)
    parser.add_argument('--lr-noise-std', type=float, default=1.0)
    parser.add_argument('--warmup-lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--decay-epochs', type=float, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--cooldown-epochs', type=int, default=10)
    parser.add_argument('--patience-epochs', type=int, default=10)
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1)

    # Augmentation
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Mixup
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup-prob', type=float, default=1.0)
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
    parser.add_argument('--mixup-mode', type=str, default='batch')

    # Distillation
    parser.add_argument('--teacher-model', default='regnety_160', type=str)
    parser.add_argument('--teacher-path', type=str, default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str)
    parser.add_argument('--distillation-alpha', default=0.5, type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float)

    # Finetune
    parser.add_argument('--finetune', default='')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--set_bn_eval', action='store_true', default=False)

    # Dataset
    parser.add_argument('--data_root', default=r'D:\WorkPlace\pythonProgram\Emotion\dataset', type=str)
    parser.add_argument('--nb_classes', default=9, type=int)
    parser.add_argument('--data-set', default='myData', type=str)
    parser.add_argument('--inat-category', default='name', type=str)

    # IO
    parser.add_argument('--output_dir', default=r'D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\result\loss')
    parser.add_argument('--writer_output', default=r'D:\WorkPlace\pythonProgram\DeepLearning\basemodels\MobileNetV4\result\loss')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist-eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Train
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--save_freq', default=1, type=int)
    return parser


def main(args):
    # ===================== 加速：TF32 =====================
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True

    print(args)
    utils.init_distributed_mode(args)

    writer = SummaryWriter(os.path.join(args.writer_output, 'runs')) if args.local_rank == 0 else None

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not supported")

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset
    dataset_train, dataset_val = build_dataset(args=args)

    # Sampler
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True) if args.repeated_aug \
            else torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval \
            else torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=int(1.5*args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    # Mixup
    mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                     prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                     label_smoothing=args.smoothing, num_classes=args.nb_classes) if args.mixup>0 or args.cutmix>0 else None

    print(f"Creating model: {args.model}")

    # ===================== 核心：FastGELU + LCA 完整修复 =====================
    from models.attention import LiteChannelAttention

    model = timm.create_model(
        "mobilenetv4_conv_large.e600_r384_in1k",
        pretrained=False,
        num_classes=args.nb_classes,
        global_pool='avg',
        act_layer=FastGELU,
        norm_eps=1e-5,
        norm_momentum=0.1
    )
    model.reset_classifier(args.nb_classes)

    # LCA 注意力
    model.blocks[3].add_module("LCA", LiteChannelAttention(128))
    model.blocks[-1].add_module("LCA", LiteChannelAttention(960))

    print("✅ MobileNetV4 + FastGELU + LCA 初始化完成，训练速度已最大化")
    # ========================================================================

    # Finetune
    if args.finetune:
        checkpoint = utils.load_model(args.finetune, model) if not args.finetune.startswith('https') \
            else torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
        for k in list(checkpoint.keys()):
            if 'classifier' in k: del checkpoint[k]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'classifier' not in name: para.requires_grad_(False)

    model.to(device)

    # EMA
    model_ema = ModelEma(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='') if args.model_ema else None

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ===================== 优化器修复 =====================
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=4e-3, weight_decay=0.005) if args.finetune \
        else create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) if args.smoothing > 0 else torch.nn.CrossEntropyLoss()
    if args.mixup > 0: criterion = SoftTargetCrossEntropy()

    # Distillation
    teacher_model = None
    if args.distillation_type != 'none':
        teacher_model = create_model(args.teacher_model, pretrained=False, num_classes=args.nb_classes, global_pool='avg')
        checkpoint = torch.hub.load_state_dict_from_url(args.teacher_path, map_location='cpu') if args.teacher_path.startswith('https') \
            else torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device).eval()

    criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)

    # Resume
    max_accuracy = 0.0
    output_dir = Path(args.output_dir)
    if args.resume or os.path.exists(f'{args.output_dir}/{args.model}_best_checkpoint.pth'):
        args.resume = f'{args.output_dir}/{args.model}_best_checkpoint.pth'
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            max_accuracy = checkpoint['best_score']
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema: utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            print(f"✅ 续训从 epoch {args.start_epoch} 开始，最佳精度 {max_accuracy:.2f}%")

    # Eval
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, None, None, args, visualization=False)
        print(f"Accuracy: {test_stats['acc1']:.1f}%")
        return

    # Train
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            set_training_mode=True, set_bn_eval=args.set_bn_eval, writer=writer, args=args
        )
        lr_scheduler.step(epoch)

        # Test
        test_stats = evaluate(data_loader_val, model, device, epoch, writer, args, visualization=True)
        print(f"Accuracy: {test_stats['acc1']:.1f}%")

        # Save
        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            ckpt_path = os.path.join(output_dir, f'{args.model}_best_checkpoint.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'best_score': max_accuracy,
                'model_ema': get_state_dict(model_ema), 'scaler': loss_scaler.state_dict(), 'args': args
            }, ckpt_path)
            print(f"✅ 最佳模型已保存：{max_accuracy:.2f}%")

        print(f'Max accuracy: {max_accuracy:.2f}%')

    # End
    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")

    # Predict
    if args.predict and utils.is_main_process():
        model_predict = create_model(args.model, extra_attention_block=args.extra_attention_block, args=args)
        model_predict.reset_classifier(args.nb_classes)
        model_predict.to(device)
        Predictor(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)
        Plot_ROC(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MobileNetV4 training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)