import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import csv
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from utils.hr_module import build_hr_module
from utils.rccr_diagnosis import RCCRDiagnosticRecorder
from utils.rccr_module import (
    GlobalRCCRCalibrator,
    RegionConditionedRCCRCalibrator,
    apply_logit_residual,
    balanced_calibration_batch,
    build_rccr_evidence,
    build_rccr_weight,
    calibration_lambda,
    compute_jsd_agreement,
    compute_region_relative_delta,
    compute_region_selective_suppress_delta,
    summarize_weight,
)
from utils.rcts_module import (
    build_fixed_rcts_route_weight,
    build_ps_wr_route_weight,
    compute_binary_region_map,
    compute_normalized_entropy,
    weak_intensity_perturb,
)
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_learning_rate(iter_num, max_iterations):
    if args.lr_schedule == 'poly':
        progress = min(float(iter_num) / float(max_iterations), 1.0)
        lr = base_lr * ((1.0 - progress) ** args.poly_power)
        return max(lr, args.min_lr)
    return base_lr

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='F:/Pycharm/Project2/semi/', help='Path that contains the data folder')
parser.add_argument('--save_path', type=str, default='F:/Pycharm/Project2/mutual-learning-with-reliable-pseudo-labels-main/runs', help='Path to save checkpoints and logs')
parser.add_argument('--exp', type=str, default='MCNet', help='exp_name')
parser.add_argument('--model', type=str, default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=50, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers; use 0 on Windows')
parser.add_argument('--amp', type=int, default=0, help='Use CUDA automatic mixed precision training')
parser.add_argument('--eval_start_iter', type=int, default=800, help='Start validation after this iteration')
parser.add_argument('--eval_interval', type=int, default=200, help='Validation interval in iterations')
parser.add_argument('--save_eval_checkpoint', type=int, default=0, help='Save a checkpoint at every validation evaluation')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'poly'], help='Learning rate schedule')
parser.add_argument('--poly_power', type=float, default=0.9, help='Power for polynomial learning rate decay')
parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate for polynomial decay')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--hr_type', type=str, default='original', choices=['original', 'region', 'region_memory', 'gated_region_memory', 'adaptive_gated_region_memory'], help='HR reliability weighting type')
parser.add_argument('--hr_boundary_width', type=int, default=1, help='Boundary width in voxels for RegionHR')
parser.add_argument('--hr_min_voxels', type=int, default=16, help='Minimum voxels required for a RegionHR prototype')
parser.add_argument('--hr_memory_weight', type=float, default=0.2, help='Historical prototype weight for RegionMemoryHR')
parser.add_argument('--hr_max_memory_weight', type=float, default=0.2, help='Maximum historical prototype weight for AdaptiveGatedRegionMemoryHR')
parser.add_argument('--hr_fusion_alpha', type=float, default=None, help='Deprecated alias of --hr_memory_weight')
parser.add_argument('--hr_memory_momentum', type=float, default=0.1, help='EMA momentum for RegionMemoryHR memory update')
parser.add_argument('--hr_gate_tau', type=float, default=0.7, help='Confidence threshold for GatedRegionMemoryHR memory update')
parser.add_argument('--hr_memory_start_iter', type=int, default=0, help='Legacy argument kept for old commands; ignored by clean E3/E4')
parser.add_argument('--rccr_diag', type=int, default=0, help='Record RCCR diagnostic evidence without changing training loss')
parser.add_argument('--rccr_diag_interval', type=int, default=50, help='Iteration interval for RCCR diagnostic sampling')
parser.add_argument('--rccr_diag_samples_per_group', type=int, default=128, help='Voxel samples per region/correctness group for RCCR diagnostics')
parser.add_argument('--rccr_type', type=str, default='none', choices=['none', 'agreement', 'global_calibrator', 'region_calibrator', 'region_selective_suppress'], help='RCCR training weight type; B2 uses agreement, B3 uses global_calibrator, B4 uses region_calibrator, B4-RC2 uses region_selective_suppress')
parser.add_argument('--rccr_agreement_floor', type=float, default=0.0, help='Minimum cross-view agreement weight for B2')
parser.add_argument('--rccr_agreement_gamma', type=float, default=1.0, help='Power applied to cross-view agreement for B2')
parser.add_argument('--rccr_log_interval', type=int, default=50, help='Iteration interval for RCCR training-weight logs')
parser.add_argument('--rccr_cal_lr', type=float, default=1e-3, help='Learning rate for B3 global RCCR calibrator')
parser.add_argument('--rccr_cal_weight_decay', type=float, default=0.0, help='Weight decay for B3 global RCCR calibrator')
parser.add_argument('--rccr_cal_samples_per_group', type=int, default=128, help='Labeled voxels sampled per region/correctness group for B3 calibrator')
parser.add_argument('--rccr_cal_start_iter', type=int, default=500, help='Iteration where B3 starts applying learned calibration')
parser.add_argument('--rccr_cal_warmup_train', type=int, default=0, help='Whether to train the RCCR calibrator before --rccr_cal_start_iter; 0 is faster and starts calibration training at start_iter')
parser.add_argument('--rccr_cal_rampup_iter', type=int, default=500, help='Iterations used to linearly ramp up learned calibration')
parser.add_argument('--rccr_cal_apply_mode', type=str, default='logit_residual', choices=['logit_residual', 'residual', 'multiply_residual', 'replace', 'region_relative'], help='How B3/B4 q affects source reliability; region_relative enables B4-RC')
parser.add_argument('--rccr_cal_logit_eps', type=float, default=1e-4, help='Clamp epsilon for B3-v2 logit residual calibration')
parser.add_argument('--rccr_cal_delta_clip', type=float, default=0.0, help='Optional absolute clamp for RCCR residual delta; 0 disables clipping')
parser.add_argument('--rccr_region_relative_scale', type=float, default=0.5, help='Maximum absolute logit residual for B4-RC region-relative calibration')
parser.add_argument('--rccr_selective_delta_scale', type=float, default=0.3, help='Maximum suppressive logit residual alpha for B4-RC2')
parser.add_argument('--rccr_selective_tail_threshold', type=float, default=0.5, help='Region-standardized low-score tail threshold tau for B4-RC2')
parser.add_argument('--rccr_cal_scale_min', type=float, default=0.5, help='Minimum B2 scale when --rccr_cal_apply_mode multiply_residual')
parser.add_argument('--rccr_cal_scale_max', type=float, default=1.5, help='Maximum B2 scale when --rccr_cal_apply_mode multiply_residual')
parser.add_argument('--wr_type', type=str, default='original', choices=['original', 'rcts_v0', 'rcts_v1', 'ps_wr', 'dtsr_stability'], help='Teacher selection mechanism; original uses entropy WR, ps_wr uses perturbation-stability WR. dtsr_stability is a deprecated alias of ps_wr')
parser.add_argument('--rcts_start_iter', type=int, default=500, help='Iteration where fixed RCTS starts replacing original WR')
parser.add_argument('--rcts_rampup_iter', type=int, default=500, help='Iterations used to linearly ramp from original WR to fixed RCTS')
parser.add_argument('--rcts_entropy_margin_inner', type=float, default=0.0, help='Minimum entropy advantage for inner-region fixed RCTS')
parser.add_argument('--rcts_entropy_margin_boundary', type=float, default=0.01, help='Minimum entropy advantage for boundary-region fixed RCTS')
parser.add_argument('--rcts_agreement_tau_inner', type=float, default=0.0, help='Minimum JSD agreement for inner-region fixed RCTS')
parser.add_argument('--rcts_agreement_tau_boundary', type=float, default=0.6, help='Minimum JSD agreement for boundary-region fixed RCTS')
parser.add_argument('--rcts_hr_weight', type=float, default=1.0, help='HR advantage coefficient for RCTS-v1')
parser.add_argument('--rcts_hr_margin_inner', type=float, default=0.0, help='Minimum HR advantage margin for inner-region RCTS-v1')
parser.add_argument('--rcts_hr_margin_boundary', type=float, default=0.0, help='Minimum HR advantage margin for boundary-region RCTS-v1')
parser.add_argument('--rcts_log_interval', type=int, default=50, help='Iteration interval for RCTS logs')
parser.add_argument('--ps_wr_margin', type=float, default=0.0, help='Minimum source-vs-target self-stability advantage for PS-WR')
parser.add_argument('--ps_wr_noise_std', type=float, default=0.03, help='Weak perturbation noise std for PS-WR')
parser.add_argument('--ps_wr_intensity_scale', type=float, default=0.05, help='Weak perturbation multiplicative scale range for PS-WR')
parser.add_argument('--ps_wr_intensity_shift', type=float, default=0.05, help='Weak perturbation additive shift range for PS-WR')
parser.add_argument('--dtsr_stability_margin', type=float, default=None, help='Deprecated alias of --ps_wr_margin')
parser.add_argument('--dtsr_noise_std', type=float, default=None, help='Deprecated alias of --ps_wr_noise_std')
parser.add_argument('--dtsr_intensity_scale', type=float, default=None, help='Deprecated alias of --ps_wr_intensity_scale')
parser.add_argument('--dtsr_intensity_shift', type=float, default=None, help='Deprecated alias of --ps_wr_intensity_shift')

args = parser.parse_args()
if args.wr_type == 'dtsr_stability':
    args.wr_type = 'ps_wr'
if args.dtsr_stability_margin is not None:
    args.ps_wr_margin = args.dtsr_stability_margin
if args.dtsr_noise_std is not None:
    args.ps_wr_noise_std = args.dtsr_noise_std
if args.dtsr_intensity_scale is not None:
    args.ps_wr_intensity_scale = args.dtsr_intensity_scale
if args.dtsr_intensity_shift is not None:
    args.ps_wr_intensity_shift = args.dtsr_intensity_shift
if args.hr_fusion_alpha is not None:
    args.hr_memory_weight = args.hr_fusion_alpha
if args.rccr_cal_apply_mode == 'residual':
    args.rccr_cal_apply_mode = 'logit_residual'
if args.rccr_cal_apply_mode == 'region_relative' and args.rccr_type != 'region_calibrator':
    raise ValueError("--rccr_cal_apply_mode region_relative requires --rccr_type region_calibrator.")
if args.rccr_type == 'region_selective_suppress' and args.rccr_cal_apply_mode != 'logit_residual':
    raise ValueError("--rccr_type region_selective_suppress expects --rccr_cal_apply_mode logit_residual.")

snapshot_path = os.path.join(args.save_path, '{}_{}_{}_labeled'.format(args.dataset_name, args.exp, args.labelnum), args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Pancreas/'
    args.max_samples = 62
elif args.dataset_name == "Spleen":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Spleen/'
    args.max_samples = 33
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    hr_module = build_hr_module(args.hr_type, num_classes=num_classes, boundary_width=args.hr_boundary_width, min_voxels=args.hr_min_voxels, memory_weight=args.hr_memory_weight, max_memory_weight=args.hr_max_memory_weight, memory_momentum=args.hr_memory_momentum, gate_tau=args.hr_gate_tau, memory_start_iter=args.hr_memory_start_iter).cuda()
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    elif args.dataset_name == "Spleen":
        db_train = Spleen(base_dir=train_data_path,
                          split='train',
                          transform=transforms.Compose([
                              RandomCrop(patch_size),
                              ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
 
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn if args.num_workers > 0 else None)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    rccr_calibrator = None
    rccr_cal_optimizer = None
    rccr_calibrator_prefix = 'rccr_global_calibrator'
    if args.rccr_type in ['global_calibrator', 'region_calibrator', 'region_selective_suppress']:
        if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
            rccr_calibrator = RegionConditionedRCCRCalibrator(input_dim=3).cuda()
            if args.rccr_type == 'region_selective_suppress':
                rccr_calibrator_prefix = 'rccr_region_selective_suppress_calibrator'
            else:
                rccr_calibrator_prefix = 'rccr_region_calibrator'
        else:
            rccr_calibrator = GlobalRCCRCalibrator(input_dim=3).cuda()
        rccr_cal_optimizer = optim.Adam(
            rccr_calibrator.parameters(),
            lr=args.rccr_cal_lr,
            weight_decay=args.rccr_cal_weight_decay,
        )
    writer = SummaryWriter(snapshot_path + '/log')
    eval_history_path = os.path.join(snapshot_path, 'eval_history.csv')
    if not os.path.exists(eval_history_path):
        with open(eval_history_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'iter', 'dice', 'best_dice', 'is_best', 'loss', 'loss_dice', 'loss_consistency', 'lr'
            ])
    rccr_recorder = None
    if args.rccr_diag:
        rccr_recorder = RCCRDiagnosticRecorder(
            snapshot_path=snapshot_path,
            num_classes=num_classes,
            interval=args.rccr_diag_interval,
            samples_per_group=args.rccr_diag_samples_per_group,
        )
        logging.info(
            "RCCR diagnosis enabled: interval %d, samples_per_group %d. This does not change training loss." % (
                args.rccr_diag_interval,
                args.rccr_diag_samples_per_group,
            )
        )
    if args.rccr_type != 'none':
        logging.info(
            "RCCR training weight enabled: type %s, agreement_floor %.4f, agreement_gamma %.4f. "
            "For B2 this uses source reliability Region-HR * detached cross-view JSD agreement." % (
                args.rccr_type,
                args.rccr_agreement_floor,
                args.rccr_agreement_gamma,
            )
        )
        if args.rccr_type in ['global_calibrator', 'region_calibrator', 'region_selective_suppress']:
            logging.info(
                "RCCR calibrator enabled: type %s, lr %.6f, samples_per_group %d, start_iter %d, rampup_iter %d, "
                "apply_mode %s, logit_eps %.6f, delta_clip %.4f, region_relative_scale %.4f, "
                "selective_delta_scale %.4f, selective_tail_threshold %.4f, scale_min %.4f, scale_max %.4f. "
                "In logit_residual mode, W=sigmoid(logit(B2)+lambda*delta); W is detached in unsupervised loss." % (
                    args.rccr_type,
                    args.rccr_cal_lr,
                    args.rccr_cal_samples_per_group,
                    args.rccr_cal_start_iter,
                    args.rccr_cal_rampup_iter,
                    args.rccr_cal_apply_mode,
                    args.rccr_cal_logit_eps,
                    args.rccr_cal_delta_clip,
                    args.rccr_region_relative_scale,
                    args.rccr_selective_delta_scale,
                    args.rccr_selective_tail_threshold,
                    args.rccr_cal_scale_min,
                    args.rccr_cal_scale_max,
                )
            )
            if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                logging.info(
                    "Region RCCR calibrator uses four source-region residual heads: "
                    "0 bg_inner, 1 bg_boundary, 2 fg_boundary, 3 fg_inner."
                )
            if args.rccr_type == 'region_selective_suppress':
                logging.info(
                    "B4-RC2 selective suppression is enabled: delta is one-sided in [-%.4f, 0], "
                    "so W never exceeds the B2 base reliability." % (
                        args.rccr_selective_delta_scale,
                    )
                )
        if args.hr_type != 'region':
            logging.warning("RCCR B2/B3/B4 is designed for --hr_type region; current hr_type is %s.", args.hr_type)
    if args.wr_type in ['rcts_v0', 'rcts_v1']:
        logging.info(
            "Fixed %s teacher selection enabled: start_iter %d, rampup_iter %d, "
            "entropy_margin inner/boundary %.4f/%.4f, agreement_tau inner/boundary %.4f/%.4f, "
            "hr_weight %.4f, hr_margin inner/boundary %.4f/%.4f. "
            "This uses no trainable teacher selector." % (
                args.wr_type,
                args.rcts_start_iter,
                args.rcts_rampup_iter,
                args.rcts_entropy_margin_inner,
                args.rcts_entropy_margin_boundary,
                args.rcts_agreement_tau_inner,
                args.rcts_agreement_tau_boundary,
                args.rcts_hr_weight,
                args.rcts_hr_margin_inner,
                args.rcts_hr_margin_boundary,
            )
        )
        logging.info(
            "Fixed RCTS uses source-pseudo-label regions: 0 bg_inner, 1 bg_boundary, 2 fg_boundary, 3 fg_inner."
        )
    if args.wr_type == 'ps_wr':
        logging.info(
            "PS-WR teacher selection enabled: start_iter %d, rampup_iter %d, "
            "stability_margin %.6f, weak perturbation noise_std %.4f, intensity_scale %.4f, intensity_shift %.4f. "
            "This keeps HR/RCCR as reliability weights and only changes WR teacher routing." % (
                args.rcts_start_iter,
                args.rcts_rampup_iter,
                args.ps_wr_margin,
                args.ps_wr_noise_std,
                args.ps_wr_intensity_scale,
                args.ps_wr_intensity_shift,
            )
        )
    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info(
        "Learning rate schedule: %s, base_lr %.8f, poly_power %.4f, min_lr %.8f" % (
            args.lr_schedule,
            base_lr,
            args.poly_power,
            args.min_lr,
        )
    )
    pixel_criterion = losses.ce_loss_mask
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.Binary_dice_loss

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            collect_rccr_stats_this_iter = (
                args.rccr_log_interval > 0
                and (iter_num + 1) % args.rccr_log_interval == 0
            )

            model.train()

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                outputs, outfeats = model(volume_batch)
            if args.amp:
                outputs = [output.float() for output in outputs]
                outfeats = [outfeat.float() for outfeat in outfeats]
            num_outputs = len(outputs)


            y_all = torch.zeros((num_outputs, ) + outputs[0].shape).cuda()

            # loss_seg = 0
            loss_seg_dice = 0
            for idx in range(num_outputs):
                y = outputs[idx]
                y_prob = F.softmax(y, dim=1)
                # loss_seg += F.cross_entropy(y_prob[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)

                y_all[idx] = y_prob

            loss_consist = 0
            pixel_consist = 0
            hr_weights_for_diag = [None for _ in range(num_outputs)]
            rcts_route_stats = []
            rcts_agreement_stats = []
            rcts_entropy_adv_stats = []
            rcts_hr_adv_stats = []
            rcts_score_stats = []
            rccr_weight_stats = []
            rccr_q_stats = []
            rccr_delta_stats = []
            rccr_raw_score_stats = []
            rccr_relative_score_stats = []
            rccr_shift_stats = []
            rccr_region_weight_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_region_q_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_region_delta_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_region_raw_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_region_relative_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_region_shift_stats = {name: [] for name in RegionConditionedRCCRCalibrator.REGION_NAMES}
            rccr_cal_feature_chunks = []
            rccr_cal_target_chunks = []
            rccr_cal_weight_chunks = []
            rccr_cal_base_chunks = []
            rccr_cal_region_chunks = []
            rccr_cal_sampled = 0
            rccr_cal_correct = 0
            rccr_cal_wrong = 0
            rccr_cal_groups = 0
            rccr_cal_candidate = 0
            rccr_cal_seen_sources = set()

            branch_pseudo_labels = []
            branch_entropy = []
            branch_hr_weights = []
            branch_region_maps = []
            branch_stability = [None for _ in range(num_outputs)]
            for branch_idx in range(num_outputs):
                branch_pseudo = torch.argmax(y_all[branch_idx], dim=1, keepdim=True)
                branch_entropy_map = compute_normalized_entropy(
                    y_all[branch_idx],
                    num_classes=num_classes,
                )
                branch_hr = hr_module(
                    outfeats[branch_idx],
                    y_all[branch_idx],
                    branch_pseudo,
                    branch_id=branch_idx,
                )
                branch_region = compute_binary_region_map(
                    branch_pseudo[:, 0].detach(),
                    boundary_width=args.hr_boundary_width,
                    spatial_size=branch_pseudo.shape[-3:],
                )
                branch_pseudo_labels.append(branch_pseudo)
                branch_entropy.append(branch_entropy_map)
                branch_hr_weights.append(branch_hr)
                branch_region_maps.append(branch_region)
                hr_weights_for_diag[branch_idx] = branch_hr.detach()

            if args.wr_type == 'ps_wr':
                with torch.no_grad():
                    perturbed_volume_batch = weak_intensity_perturb(
                        volume_batch,
                        noise_std=args.ps_wr_noise_std,
                        intensity_scale=args.ps_wr_intensity_scale,
                        intensity_shift=args.ps_wr_intensity_shift,
                    )
                    was_training = model.training
                    model.eval()
                    with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                        perturb_outputs, _ = model(perturbed_volume_batch)
                    if was_training:
                        model.train()
                    if args.amp:
                        perturb_outputs = [output.float() for output in perturb_outputs]
                    for branch_idx in range(num_outputs):
                        perturb_prob = F.softmax(perturb_outputs[branch_idx], dim=1)
                        branch_stability[branch_idx] = compute_jsd_agreement(
                            y_all[branch_idx].detach(),
                            perturb_prob.detach(),
                            num_classes=num_classes,
                        )

            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        uncertainty_o1 = branch_entropy[i]
                        uncertainty_o2 = branch_entropy[j]
                        mask = (uncertainty_o1 > uncertainty_o2).float()

                        pseudo_labels = branch_pseudo_labels[j]
                        hr_weight = branch_hr_weights[j]
                        target_hr_weight = branch_hr_weights[i]
                        source_region_map = branch_region_maps[j]
                        route_weight = mask
                        if args.wr_type in ['rcts_v0', 'rcts_v1']:
                            with torch.no_grad():
                                rcts_raw_route, rcts_stats = build_fixed_rcts_route_weight(
                                    args.wr_type,
                                    source_prob=y_all[j],
                                    target_prob=y_all[i],
                                    source_hr=hr_weight,
                                    target_hr=target_hr_weight,
                                    source_entropy=branch_entropy[j],
                                    target_entropy=branch_entropy[i],
                                    source_region_map=source_region_map,
                                    num_classes=num_classes,
                                    entropy_margin_inner=args.rcts_entropy_margin_inner,
                                    entropy_margin_boundary=args.rcts_entropy_margin_boundary,
                                    agreement_tau_inner=args.rcts_agreement_tau_inner,
                                    agreement_tau_boundary=args.rcts_agreement_tau_boundary,
                                    hr_margin_inner=args.rcts_hr_margin_inner,
                                    hr_margin_boundary=args.rcts_hr_margin_boundary,
                                    hr_weight=args.rcts_hr_weight,
                                )
                                rcts_lambda = calibration_lambda(
                                    iter_num + 1,
                                    start_iter=args.rcts_start_iter,
                                    rampup_iter=args.rcts_rampup_iter,
                                )
                                route_weight = (
                                    (1.0 - rcts_lambda) * mask
                                    + rcts_lambda * rcts_raw_route
                                ).clamp(min=0.0, max=1.0)
                            rcts_route_stats.append(summarize_weight(route_weight))
                            rcts_agreement_stats.append(summarize_weight(rcts_stats["agreement"]))
                            rcts_entropy_adv_stats.append(summarize_weight(rcts_stats["entropy_advantage"]))
                            rcts_hr_adv_stats.append(summarize_weight(rcts_stats["hr_advantage"]))
                            rcts_score_stats.append(summarize_weight(rcts_stats["teacher_score"]))
                        elif args.wr_type == 'ps_wr':
                            with torch.no_grad():
                                ps_wr_raw_route, ps_wr_stats = build_ps_wr_route_weight(
                                    source_stability=branch_stability[j],
                                    target_stability=branch_stability[i],
                                    stability_margin=args.ps_wr_margin,
                                )
                                ps_wr_lambda = calibration_lambda(
                                    iter_num + 1,
                                    start_iter=args.rcts_start_iter,
                                    rampup_iter=args.rcts_rampup_iter,
                                )
                                route_weight = (
                                    (1.0 - ps_wr_lambda) * mask
                                    + ps_wr_lambda * ps_wr_raw_route
                                ).clamp(min=0.0, max=1.0)
                            rcts_route_stats.append(summarize_weight(route_weight))
                            rcts_entropy_adv_stats.append(summarize_weight(ps_wr_stats["stability_advantage"]))
                            rcts_score_stats.append(summarize_weight(ps_wr_stats["teacher_score"]))

                        weight_pixel_t = hr_weight * route_weight
                        if args.rccr_type == 'agreement':
                            rccr_weight = build_rccr_weight(
                                args.rccr_type,
                                source_prob=y_all[j],
                                target_prob=y_all[i],
                                num_classes=num_classes,
                                agreement_floor=args.rccr_agreement_floor,
                                agreement_gamma=args.rccr_agreement_gamma,
                            )
                            rccr_weight_stats.append(summarize_weight(rccr_weight))
                            weight_pixel_t = weight_pixel_t * rccr_weight.detach()
                        elif args.rccr_type in ['global_calibrator', 'region_calibrator', 'region_selective_suppress']:
                            evidence, agreement, uncertainty_reliability = build_rccr_evidence(
                                source_prob=y_all[j],
                                target_prob=y_all[i],
                                hr_weight=hr_weight,
                                num_classes=num_classes,
                            )
                            b2_agreement = agreement.detach()
                            if args.rccr_agreement_gamma != 1.0:
                                b2_agreement = b2_agreement.clamp(min=0.0, max=1.0).pow(args.rccr_agreement_gamma)
                            agreement_floor = float(args.rccr_agreement_floor)
                            if agreement_floor < 0.0 or agreement_floor > 1.0:
                                raise ValueError("agreement_floor must be in [0, 1]. Got {}.".format(agreement_floor))
                            b2_agreement = agreement_floor + (1.0 - agreement_floor) * b2_agreement
                            b2_reliability = hr_weight.detach() * b2_agreement
                            if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                                source_region_map = branch_region_maps[j]
                            b3_lambda = calibration_lambda(
                                iter_num + 1,
                                start_iter=args.rccr_cal_start_iter,
                                rampup_iter=args.rccr_cal_rampup_iter,
                            )
                            with torch.no_grad():
                                if b3_lambda <= 0.0:
                                    source_reliability = b2_reliability
                                    q_weight = b2_reliability
                                    delta_weight = torch.zeros_like(b2_reliability)
                                    raw_delta_weight = delta_weight
                                    relative_score_weight = delta_weight
                                else:
                                    evidence_flat = evidence.reshape(-1, evidence.shape[-1])
                                    if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                                        region_flat = source_region_map.reshape(-1)
                                        raw_delta_weight = rccr_calibrator.forward_delta(evidence_flat, region_flat).view_as(hr_weight)
                                    else:
                                        raw_delta_weight = rccr_calibrator.forward_delta(evidence_flat).view_as(hr_weight)
                                    relative_score_weight = None
                                    if args.rccr_type == 'region_selective_suppress':
                                        relative_score_weight, delta_weight = compute_region_selective_suppress_delta(
                                            raw_delta_weight,
                                            source_region_map,
                                            suppress_scale=args.rccr_selective_delta_scale,
                                            tail_threshold=args.rccr_selective_tail_threshold,
                                            min_voxels=args.hr_min_voxels,
                                            num_regions=len(RegionConditionedRCCRCalibrator.REGION_NAMES),
                                        )
                                    elif args.rccr_cal_apply_mode == 'region_relative':
                                        relative_score_weight, delta_weight = compute_region_relative_delta(
                                            raw_delta_weight,
                                            source_region_map,
                                            max_delta=args.rccr_region_relative_scale,
                                            num_regions=len(RegionConditionedRCCRCalibrator.REGION_NAMES),
                                        )
                                    else:
                                        delta_weight = raw_delta_weight
                                        if args.rccr_cal_delta_clip > 0:
                                            delta_weight = delta_weight.clamp(
                                                min=-float(args.rccr_cal_delta_clip),
                                                max=float(args.rccr_cal_delta_clip),
                                            )
                                    if args.rccr_cal_apply_mode == 'replace':
                                        if args.rccr_type == 'region_calibrator':
                                            q_weight = rccr_calibrator(evidence_flat, region_flat).view_as(hr_weight)
                                        else:
                                            q_weight = rccr_calibrator(evidence_flat).view_as(hr_weight)
                                        calibrated_reliability = q_weight
                                    elif args.rccr_cal_apply_mode == 'logit_residual':
                                        q_weight = apply_logit_residual(
                                            b2_reliability,
                                            delta_weight,
                                            residual_lambda=1.0,
                                            eps=args.rccr_cal_logit_eps,
                                        )
                                        calibrated_reliability = apply_logit_residual(
                                            b2_reliability,
                                            delta_weight,
                                            residual_lambda=b3_lambda,
                                            eps=args.rccr_cal_logit_eps,
                                        )
                                    elif args.rccr_cal_apply_mode == 'region_relative':
                                        q_weight = apply_logit_residual(
                                            b2_reliability,
                                            delta_weight,
                                            residual_lambda=1.0,
                                            eps=args.rccr_cal_logit_eps,
                                        )
                                        calibrated_reliability = apply_logit_residual(
                                            b2_reliability,
                                            delta_weight,
                                            residual_lambda=b3_lambda,
                                            eps=args.rccr_cal_logit_eps,
                                        )
                                    else:
                                        if args.rccr_type == 'region_calibrator':
                                            q_weight = rccr_calibrator(evidence_flat, region_flat).view_as(hr_weight)
                                        else:
                                            q_weight = rccr_calibrator(evidence_flat).view_as(hr_weight)
                                        cal_scale_min = float(args.rccr_cal_scale_min)
                                        cal_scale_max = float(args.rccr_cal_scale_max)
                                        if cal_scale_min < 0.0 or cal_scale_max < cal_scale_min:
                                            raise ValueError(
                                                "--rccr_cal_scale_min/max must satisfy 0 <= min <= max."
                                            )
                                        cal_scale = cal_scale_min + (cal_scale_max - cal_scale_min) * q_weight
                                        calibrated_reliability = b2_reliability * cal_scale
                                    if args.rccr_cal_apply_mode in ['logit_residual', 'region_relative']:
                                        source_reliability = calibrated_reliability.clamp(min=0.0, max=1.0)
                                    else:
                                        source_reliability = (
                                            (1.0 - b3_lambda) * b2_reliability
                                            + b3_lambda * calibrated_reliability
                                        ).clamp(min=0.0, max=1.0)
                                    if args.rccr_type == 'region_selective_suppress':
                                        q_weight = torch.minimum(q_weight, b2_reliability)
                                        source_reliability = torch.minimum(source_reliability, b2_reliability)
                            if collect_rccr_stats_this_iter:
                                rccr_weight_stats.append(summarize_weight(source_reliability))
                                rccr_q_stats.append(summarize_weight(q_weight))
                                rccr_delta_stats.append(summarize_weight(delta_weight))
                                if args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress':
                                    rccr_raw_score_stats.append(summarize_weight(raw_delta_weight))
                                    rccr_relative_score_stats.append(summarize_weight(relative_score_weight))
                                    rccr_shift_stats.append(summarize_weight(source_reliability - b2_reliability))
                                if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                                    for region_id, region_name in enumerate(RegionConditionedRCCRCalibrator.REGION_NAMES):
                                        region_mask = source_region_map == region_id
                                        if bool(region_mask.any().item()):
                                            rccr_region_weight_stats[region_name].append(
                                                summarize_weight(source_reliability[region_mask])
                                            )
                                            rccr_region_q_stats[region_name].append(
                                                summarize_weight(q_weight[region_mask])
                                            )
                                            rccr_region_delta_stats[region_name].append(
                                                summarize_weight(delta_weight[region_mask])
                                            )
                                            if args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress':
                                                rccr_region_raw_stats[region_name].append(
                                                    summarize_weight(raw_delta_weight[region_mask])
                                                )
                                                rccr_region_relative_stats[region_name].append(
                                                    summarize_weight(relative_score_weight[region_mask])
                                                )
                                                rccr_region_shift_stats[region_name].append(
                                                    summarize_weight((source_reliability - b2_reliability)[region_mask])
                                                )
                            weight_pixel_t = route_weight * source_reliability.detach()

                            train_rccr_calibrator_this_iter = (
                                bool(args.rccr_cal_warmup_train)
                                or (iter_num + 1) >= args.rccr_cal_start_iter
                            )
                            if train_rccr_calibrator_this_iter and j not in rccr_cal_seen_sources:
                                rccr_cal_seen_sources.add(j)
                                pseudo_full = pseudo_labels[:, 0].detach()
                                labeled_gt = label_batch[:labeled_bs].detach()
                                labeled_pseudo = pseudo_full[:labeled_bs]
                                if tuple(labeled_gt.shape[-3:]) != tuple(labeled_pseudo.shape[-3:]):
                                    labeled_gt = F.interpolate(
                                        labeled_gt.unsqueeze(1).float(),
                                        size=labeled_pseudo.shape[-3:],
                                        mode='nearest',
                                    ).squeeze(1).long()
                                if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                                    region_map = source_region_map[:labeled_bs]
                                else:
                                    region_map = branch_region_maps[j][:labeled_bs]
                                cal_features, cal_targets, cal_sample_weights, cal_base_weights, cal_region_ids, cal_stats = balanced_calibration_batch(
                                    evidence[:labeled_bs],
                                    labeled_pseudo,
                                    labeled_gt.long(),
                                    region_map,
                                    base_weight=b2_reliability[:labeled_bs],
                                    samples_per_group=args.rccr_cal_samples_per_group,
                                )
                                if cal_features is not None:
                                    rccr_cal_feature_chunks.append(cal_features)
                                    rccr_cal_target_chunks.append(cal_targets)
                                    rccr_cal_weight_chunks.append(cal_sample_weights)
                                    rccr_cal_base_chunks.append(cal_base_weights)
                                    rccr_cal_region_chunks.append(cal_region_ids)
                                rccr_cal_sampled += cal_stats["sampled"]
                                rccr_cal_correct += cal_stats["correct"]
                                rccr_cal_wrong += cal_stats["wrong"]
                                rccr_cal_groups += cal_stats["groups"]
                                rccr_cal_candidate += cal_stats["candidate"]

                        loss_t = consistency_criterion(outputs[i], torch.argmax(y_all[j], dim=1).detach())
                        loss_consist += (loss_t * weight_pixel_t.detach()).sum() / (route_weight.detach().sum() + 1e-6)

            iter_num = iter_num + 1
            if rccr_recorder is not None:
                rccr_recorder.record(
                    iter_num=iter_num,
                    probabilities_all=y_all.detach(),
                    hr_weights=hr_weights_for_diag,
                    label_batch=label_batch.detach(),
                    labeled_bs=labeled_bs,
                    hr_module=hr_module,
                )
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = args.lamda * loss_seg_dice + consistency_weight * (loss_consist)
            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_ = get_current_learning_rate(iter_num, max_iterations)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            writer.add_scalar('LR/lr', lr_, iter_num)

            if rccr_calibrator is not None and rccr_cal_feature_chunks:
                cal_features = torch.cat(rccr_cal_feature_chunks, dim=0)
                cal_targets = torch.cat(rccr_cal_target_chunks, dim=0)
                cal_sample_weights = torch.cat(rccr_cal_weight_chunks, dim=0)
                cal_base_weights = torch.cat(rccr_cal_base_chunks, dim=0)
                cal_region_ids = torch.cat(rccr_cal_region_chunks, dim=0) if rccr_cal_region_chunks else None
                rccr_cal_optimizer.zero_grad()
                if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                    cal_delta = rccr_calibrator.forward_delta(cal_features, cal_region_ids)
                else:
                    cal_delta = rccr_calibrator.forward_delta(cal_features)
                if args.rccr_cal_delta_clip > 0 and args.rccr_cal_apply_mode != 'region_relative' and args.rccr_type != 'region_selective_suppress':
                    cal_delta = cal_delta.clamp(
                        min=-float(args.rccr_cal_delta_clip),
                        max=float(args.rccr_cal_delta_clip),
                    )
                if args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress':
                    cal_pred = torch.sigmoid(cal_delta)
                    cal_loss_raw = F.binary_cross_entropy_with_logits(cal_delta, cal_targets, reduction='none')
                elif args.rccr_cal_apply_mode == 'logit_residual':
                    cal_pred = apply_logit_residual(
                        cal_base_weights,
                        cal_delta,
                        residual_lambda=1.0,
                        eps=args.rccr_cal_logit_eps,
                    )
                    cal_loss_raw = F.binary_cross_entropy(cal_pred, cal_targets, reduction='none')
                else:
                    cal_pred = torch.sigmoid(cal_delta)
                    cal_loss_raw = F.binary_cross_entropy(cal_pred, cal_targets, reduction='none')
                loss_cal = (
                    cal_loss_raw * cal_sample_weights
                ).sum() / (cal_sample_weights.sum() + 1e-6)
                loss_cal.backward()
                rccr_cal_optimizer.step()

                if iter_num % max(args.rccr_log_interval, 1) == 0:
                    with torch.no_grad():
                        if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                            cal_delta_after = rccr_calibrator.forward_delta(cal_features, cal_region_ids)
                        else:
                            cal_delta_after = rccr_calibrator.forward_delta(cal_features)
                        if args.rccr_cal_delta_clip > 0 and args.rccr_cal_apply_mode != 'region_relative' and args.rccr_type != 'region_selective_suppress':
                            cal_delta_after = cal_delta_after.clamp(
                                min=-float(args.rccr_cal_delta_clip),
                                max=float(args.rccr_cal_delta_clip),
                            )
                        if args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress':
                            cal_pred_after = torch.sigmoid(cal_delta_after)
                        elif args.rccr_cal_apply_mode == 'logit_residual':
                            cal_pred_after = apply_logit_residual(
                                cal_base_weights,
                                cal_delta_after,
                                residual_lambda=1.0,
                                eps=args.rccr_cal_logit_eps,
                            )
                        else:
                            cal_pred_after = torch.sigmoid(cal_delta_after)
                        correct_mask = cal_targets > 0.5
                        wrong_mask = ~correct_mask
                        q_correct_mean = (
                            cal_pred_after[correct_mask].mean().item()
                            if bool(correct_mask.any().item()) else 0.0
                        )
                        q_wrong_mean = (
                            cal_pred_after[wrong_mask].mean().item()
                            if bool(wrong_mask.any().item()) else 0.0
                        )
                        delta_mean = cal_delta_after.mean().item()
                        delta_std = cal_delta_after.std(unbiased=False).item()
                        if args.rccr_type == 'global_calibrator':
                            cal_weight = rccr_calibrator.linear.weight.detach().view(-1)
                            cal_bias = rccr_calibrator.linear.bias.detach()[0]
                        else:
                            cal_weight = None
                            cal_bias = None
                    b3_lambda_for_log = calibration_lambda(
                        iter_num,
                        start_iter=args.rccr_cal_start_iter,
                        rampup_iter=args.rccr_cal_rampup_iter,
                    )
                    if args.rccr_type == 'global_calibrator':
                        logging.info(
                            'rccr cal iter %d : lambda %.6f loss %.6f sampled %d candidate %d correct %d wrong %d groups %d '
                            'q_correct %.6f q_wrong %.6f delta_mean %.6f delta_std %.6f coef_h %.6f coef_agreement %.6f coef_uncertainty_rel %.6f bias %.6f' % (
                                iter_num,
                                b3_lambda_for_log,
                                float(loss_cal.item()),
                                rccr_cal_sampled,
                                rccr_cal_candidate,
                                rccr_cal_correct,
                                rccr_cal_wrong,
                                rccr_cal_groups,
                                q_correct_mean,
                                q_wrong_mean,
                                delta_mean,
                                delta_std,
                                float(cal_weight[0].item()),
                                float(cal_weight[1].item()),
                                float(cal_weight[2].item()),
                                float(cal_bias.item()),
                            )
                        )
                    else:
                        logging.info(
                            'rccr cal iter %d : lambda %.6f loss %.6f sampled %d candidate %d correct %d wrong %d groups %d '
                            'q_correct %.6f q_wrong %.6f delta_mean %.6f delta_std %.6f mode %s' % (
                                iter_num,
                                b3_lambda_for_log,
                                float(loss_cal.item()),
                                rccr_cal_sampled,
                                rccr_cal_candidate,
                                rccr_cal_correct,
                                rccr_cal_wrong,
                                rccr_cal_groups,
                                q_correct_mean,
                                q_wrong_mean,
                                delta_mean,
                                delta_std,
                                args.rccr_type,
                            )
                        )
                        for head_info in rccr_calibrator.head_parameters():
                            region_id = head_info["region_id"]
                            region_name = head_info["region_name"]
                            region_mask = cal_region_ids == region_id
                            if bool(region_mask.any().item()):
                                region_targets = cal_targets[region_mask]
                                region_pred = cal_pred_after[region_mask]
                                region_delta = cal_delta_after[region_mask]
                                region_correct_mask = region_targets > 0.5
                                region_wrong_mask = ~region_correct_mask
                                region_q_correct = (
                                    region_pred[region_correct_mask].mean().item()
                                    if bool(region_correct_mask.any().item()) else 0.0
                                )
                                region_q_wrong = (
                                    region_pred[region_wrong_mask].mean().item()
                                    if bool(region_wrong_mask.any().item()) else 0.0
                                )
                                region_delta_mean = region_delta.mean().item()
                                region_delta_std = region_delta.std(unbiased=False).item()
                                region_correct = int(region_correct_mask.sum().item())
                                region_wrong = int(region_wrong_mask.sum().item())
                            else:
                                region_q_correct = 0.0
                                region_q_wrong = 0.0
                                region_delta_mean = 0.0
                                region_delta_std = 0.0
                                region_correct = 0
                                region_wrong = 0
                            head_weight = head_info["weight"]
                            head_bias = head_info["bias"]
                            logging.info(
                                'rccr region cal iter %d : %s sampled %d correct %d wrong %d '
                                'q_correct %.6f q_wrong %.6f delta_mean %.6f delta_std %.6f '
                                'coef_h %.6f coef_agreement %.6f coef_uncertainty_rel %.6f bias %.6f' % (
                                    iter_num,
                                    region_name,
                                    int(region_mask.sum().item()) if bool(region_mask.any().item()) else 0,
                                    region_correct,
                                    region_wrong,
                                    region_q_correct,
                                    region_q_wrong,
                                    region_delta_mean,
                                    region_delta_std,
                                    float(head_weight[0].item()),
                                    float(head_weight[1].item()),
                                    float(head_weight[2].item()),
                                    float(head_bias.item()),
                                )
                            )
                    writer.add_scalar('RCCR/cal_loss', loss_cal.item(), iter_num)
                    writer.add_scalar('RCCR/cal_lambda', b3_lambda_for_log, iter_num)
                    writer.add_scalar('RCCR/q_correct_mean', q_correct_mean, iter_num)
                    writer.add_scalar('RCCR/q_wrong_mean', q_wrong_mean, iter_num)
                    writer.add_scalar('RCCR/delta_mean', delta_mean, iter_num)
                    writer.add_scalar('RCCR/delta_std', delta_std, iter_num)
            if args.hr_type in ['gated_region_memory', 'adaptive_gated_region_memory']:
                hr_module.update_memory(y_all.detach())
                gate_stats = hr_module.pop_update_summary() if hasattr(hr_module, 'pop_update_summary') else None
                if gate_stats is not None and iter_num % 50 == 0:
                    logging.info('gate iter %d : agreement %.6f, conf_a %.6f, conf_b %.6f, confidence %.6f, pass %.6f, updates %d, rejects %d, raw_voxels %.1f, gated_voxels %.1f' % (
                        iter_num,
                        gate_stats.get('agreement_ratio', 0.0),
                        gate_stats.get('conf_pass_ratio_a', 0.0),
                        gate_stats.get('conf_pass_ratio_b', 0.0),
                        gate_stats.get('confidence_ratio', 0.0),
                        gate_stats.get('gate_pass_ratio', 0.0),
                        gate_stats.get('memory_update_count', 0),
                        gate_stats.get('memory_reject_count', 0),
                        gate_stats.get('raw_region_voxels', 0.0),
                        gate_stats.get('gated_region_voxels', 0.0),
                    ))
                    logging.info('gate conf iter %d : A min %.6f mean %.6f max %.6f | B min %.6f mean %.6f max %.6f' % (
                        iter_num,
                        gate_stats.get('conf_a_min', 0.0),
                        gate_stats.get('conf_a_mean', 0.0),
                        gate_stats.get('conf_a_max', 0.0),
                        gate_stats.get('conf_b_min', 0.0),
                        gate_stats.get('conf_b_mean', 0.0),
                        gate_stats.get('conf_b_max', 0.0),
                    ))
                    for region_key, region_stat in sorted(gate_stats.get('region_stats', {}).items()):
                        logging.info('gate region iter %d : %s region %.1f gated %.1f ratio %.6f valid %d update %d reject %d memory_valid %d' % (
                            iter_num,
                            region_key,
                            region_stat.get('region_count', 0.0),
                            region_stat.get('gated_count', 0.0),
                            region_stat.get('gate_ratio', 0.0),
                            region_stat.get('region_valid', 0),
                            region_stat.get('update_allowed', 0),
                            region_stat.get('update_rejected', 0),
                            region_stat.get('memory_valid', 0),
                        ))
                    for region_key, adaptive_stat in sorted(gate_stats.get('adaptive_stats', {}).items()):
                        sim_stat = adaptive_stat.get('similarity', {})
                        weight_stat = adaptive_stat.get('weight', {})
                        logging.info('adaptive region iter %d : %s sim mean %.6f min %.6f max %.6f zero %.6f count %d | weight mean %.6f min %.6f max %.6f zero %.6f count %d' % (
                            iter_num,
                            region_key,
                            sim_stat.get('mean', 0.0),
                            sim_stat.get('min', 0.0),
                            sim_stat.get('max', 0.0),
                            sim_stat.get('zero_ratio', 0.0),
                            sim_stat.get('count', 0),
                            weight_stat.get('mean', 0.0),
                            weight_stat.get('min', 0.0),
                            weight_stat.get('max', 0.0),
                            weight_stat.get('zero_ratio', 0.0),
                            weight_stat.get('count', 0),
                        ))
            else:
                hr_module.update_memory()
            if args.wr_type in ['rcts_v0', 'rcts_v1'] and rcts_route_stats and iter_num % max(args.rcts_log_interval, 1) == 0:
                rcts_lambda_for_log = calibration_lambda(
                    iter_num,
                    start_iter=args.rcts_start_iter,
                    rampup_iter=args.rcts_rampup_iter,
                )
                logging.info(
                    'rcts fixed route iter %d : type %s lambda %.6f route mean %.6f min %.6f max %.6f std %.6f | '
                    'agreement mean %.6f | entropy_adv mean %.6f | hr_adv mean %.6f | score mean %.6f' % (
                        iter_num,
                        args.wr_type,
                        rcts_lambda_for_log,
                        float(np.mean([s["mean"] for s in rcts_route_stats])),
                        float(np.min([s["min"] for s in rcts_route_stats])),
                        float(np.max([s["max"] for s in rcts_route_stats])),
                        float(np.mean([s["std"] for s in rcts_route_stats])),
                        float(np.mean([s["mean"] for s in rcts_agreement_stats])),
                        float(np.mean([s["mean"] for s in rcts_entropy_adv_stats])),
                        float(np.mean([s["mean"] for s in rcts_hr_adv_stats])),
                        float(np.mean([s["mean"] for s in rcts_score_stats])),
                    )
                )
            if args.wr_type == 'ps_wr' and rcts_route_stats and iter_num % max(args.rcts_log_interval, 1) == 0:
                ps_wr_lambda_for_log = calibration_lambda(
                    iter_num,
                    start_iter=args.rcts_start_iter,
                    rampup_iter=args.rcts_rampup_iter,
                )
                logging.info(
                    'ps-wr route iter %d : lambda %.6f route mean %.6f min %.6f max %.6f std %.6f | '
                    'stability_adv mean %.6f | score mean %.6f' % (
                        iter_num,
                        ps_wr_lambda_for_log,
                        float(np.mean([s["mean"] for s in rcts_route_stats])),
                        float(np.min([s["min"] for s in rcts_route_stats])),
                        float(np.max([s["max"] for s in rcts_route_stats])),
                        float(np.mean([s["std"] for s in rcts_route_stats])),
                        float(np.mean([s["mean"] for s in rcts_entropy_adv_stats])),
                        float(np.mean([s["mean"] for s in rcts_score_stats])),
                    )
                )
            if args.rccr_type != 'none' and rccr_weight_stats and iter_num % max(args.rccr_log_interval, 1) == 0:
                logging.info(
                    'rccr train iter %d : type %s mean %.6f min %.6f max %.6f std %.6f' % (
                        iter_num,
                        args.rccr_type,
                        float(np.mean([s["mean"] for s in rccr_weight_stats])),
                        float(np.min([s["min"] for s in rccr_weight_stats])),
                        float(np.max([s["max"] for s in rccr_weight_stats])),
                        float(np.mean([s["std"] for s in rccr_weight_stats])),
                    )
                )
                if rccr_q_stats:
                    logging.info(
                        'rccr q iter %d : mean %.6f min %.6f max %.6f std %.6f' % (
                            iter_num,
                            float(np.mean([s["mean"] for s in rccr_q_stats])),
                            float(np.min([s["min"] for s in rccr_q_stats])),
                            float(np.max([s["max"] for s in rccr_q_stats])),
                            float(np.mean([s["std"] for s in rccr_q_stats])),
                        )
                    )
                if rccr_delta_stats:
                    logging.info(
                        'rccr delta iter %d : mean %.6f min %.6f max %.6f std %.6f' % (
                            iter_num,
                            float(np.mean([s["mean"] for s in rccr_delta_stats])),
                            float(np.min([s["min"] for s in rccr_delta_stats])),
                            float(np.max([s["max"] for s in rccr_delta_stats])),
                            float(np.mean([s["std"] for s in rccr_delta_stats])),
                        )
                    )
                if (args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress') and rccr_raw_score_stats:
                    logging.info(
                        'rccr rc iter %d : raw mean %.6f min %.6f max %.6f std %.6f | '
                        'relative mean %.6f min %.6f max %.6f std %.6f | '
                        'W_minus_w0 mean %.6f min %.6f max %.6f std %.6f' % (
                            iter_num,
                            float(np.mean([s["mean"] for s in rccr_raw_score_stats])),
                            float(np.min([s["min"] for s in rccr_raw_score_stats])),
                            float(np.max([s["max"] for s in rccr_raw_score_stats])),
                            float(np.mean([s["std"] for s in rccr_raw_score_stats])),
                            float(np.mean([s["mean"] for s in rccr_relative_score_stats])),
                            float(np.min([s["min"] for s in rccr_relative_score_stats])),
                            float(np.max([s["max"] for s in rccr_relative_score_stats])),
                            float(np.mean([s["std"] for s in rccr_relative_score_stats])),
                            float(np.mean([s["mean"] for s in rccr_shift_stats])),
                            float(np.min([s["min"] for s in rccr_shift_stats])),
                            float(np.max([s["max"] for s in rccr_shift_stats])),
                            float(np.mean([s["std"] for s in rccr_shift_stats])),
                        )
                    )
                if args.rccr_type in ['region_calibrator', 'region_selective_suppress']:
                    for region_name in RegionConditionedRCCRCalibrator.REGION_NAMES:
                        weight_stats = rccr_region_weight_stats.get(region_name, [])
                        q_stats = rccr_region_q_stats.get(region_name, [])
                        delta_stats = rccr_region_delta_stats.get(region_name, [])
                        if not weight_stats:
                            continue
                        logging.info(
                            'rccr region train iter %d : %s W mean %.6f min %.6f max %.6f std %.6f | '
                            'q mean %.6f min %.6f max %.6f std %.6f | '
                            'delta mean %.6f min %.6f max %.6f std %.6f' % (
                                iter_num,
                                region_name,
                                float(np.mean([s["mean"] for s in weight_stats])),
                                float(np.min([s["min"] for s in weight_stats])),
                                float(np.max([s["max"] for s in weight_stats])),
                                float(np.mean([s["std"] for s in weight_stats])),
                                float(np.mean([s["mean"] for s in q_stats])),
                                float(np.min([s["min"] for s in q_stats])),
                                float(np.max([s["max"] for s in q_stats])),
                                float(np.mean([s["std"] for s in q_stats])),
                                float(np.mean([s["mean"] for s in delta_stats])),
                                float(np.min([s["min"] for s in delta_stats])),
                                float(np.max([s["max"] for s in delta_stats])),
                                float(np.mean([s["std"] for s in delta_stats])),
                            )
                        )
                        if args.rccr_cal_apply_mode == 'region_relative' or args.rccr_type == 'region_selective_suppress':
                            raw_stats = rccr_region_raw_stats.get(region_name, [])
                            relative_stats = rccr_region_relative_stats.get(region_name, [])
                            shift_stats = rccr_region_shift_stats.get(region_name, [])
                            if raw_stats:
                                logging.info(
                                    'rccr region rc iter %d : %s raw mean %.6f min %.6f max %.6f std %.6f | '
                                    'relative mean %.6f min %.6f max %.6f std %.6f | '
                                    'W_minus_w0 mean %.6f min %.6f max %.6f std %.6f' % (
                                        iter_num,
                                        region_name,
                                        float(np.mean([s["mean"] for s in raw_stats])),
                                        float(np.min([s["min"] for s in raw_stats])),
                                        float(np.max([s["max"] for s in raw_stats])),
                                        float(np.mean([s["std"] for s in raw_stats])),
                                        float(np.mean([s["mean"] for s in relative_stats])),
                                        float(np.min([s["min"] for s in relative_stats])),
                                        float(np.max([s["max"] for s in relative_stats])),
                                        float(np.mean([s["std"] for s in relative_stats])),
                                        float(np.mean([s["mean"] for s in shift_stats])),
                                        float(np.min([s["min"] for s in shift_stats])),
                                        float(np.max([s["max"] for s in shift_stats])),
                                        float(np.mean([s["std"] for s in shift_stats])),
                                    )
                                )
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' %
                         (iter_num, loss.item(), loss_seg_dice.item(), loss_consist.item()))

            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)

            if iter_num >= args.eval_start_iter and iter_num % max(args.eval_interval, 1) == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=18,
                                                          stride_z=4,
                                                          dataset_name='LA',
                                                          root_path=train_data_path)
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT',
                                                          root_path=train_data_path)
                elif args.dataset_name == "Spleen":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Spleen',
                                                          root_path=train_data_path)
                is_best = dice_sample > best_dice
                if is_best:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    if rccr_calibrator is not None:
                        torch.save(
                            rccr_calibrator.state_dict(),
                            os.path.join(snapshot_path, '{}_best.pth'.format(rccr_calibrator_prefix)),
                        )
                    logging.info("save best model to {}".format(save_mode_path))
                with open(eval_history_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        iter_num,
                        float(dice_sample),
                        float(best_dice),
                        int(is_best),
                        float(loss.item()),
                        float(loss_seg_dice.item()),
                        float(loss_consist.item()),
                        float(lr_),
                    ])
                logging.info('validation iter %d : dice %.6f, best_dice %.6f, is_best %d, lr %.8f' % (
                    iter_num, float(dice_sample), float(best_dice), int(is_best), float(lr_)
                ))
                if args.save_eval_checkpoint:
                    save_eval_path = os.path.join(snapshot_path, 'iter_{}_eval_dice_{}.pth'.format(iter_num, dice_sample))
                    torch.save(model.state_dict(), save_eval_path)
                    if rccr_calibrator is not None:
                        torch.save(
                            rccr_calibrator.state_dict(),
                            os.path.join(snapshot_path, '{}_iter_{}_eval.pth'.format(rccr_calibrator_prefix, iter_num)),
                        )
                    logging.info("save eval model to {}".format(save_eval_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                if rccr_calibrator is not None:
                    torch.save(
                        rccr_calibrator.state_dict(),
                        os.path.join(snapshot_path, '{}_iter_{}.pth'.format(rccr_calibrator_prefix, iter_num)),
                    )
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break

    writer.close()
    if rccr_recorder is not None:
        rccr_recorder.close()





