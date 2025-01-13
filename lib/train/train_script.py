import copy
import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.avtrack import build_avtrack
# forward propagation related
from lib.train.actors import AVTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "avtrack":
        # ===== distillation ====
        is_distill_training = cfg.MODEL['IS_DISTILL']
        if is_distill_training:
            cfg_teacher = copy.deepcopy(cfg)
            cfg_teacher.MODEL['BACKBONE']['TYPE'] = 'deit_tiny_patch16_224'
            cfg_teacher['MODEL']['BACKBONE']['TYPE'] = 'deit_tiny_patch16_224'
            net_teacher = build_avtrack(cfg_teacher)
            cfg_teacher.MODEL['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224'
            cfg_teacher['MODEL']['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224'
            net_teacher2 = build_avtrack(cfg_teacher)
            cfg_teacher.DATA['SEARCH']['SIZE'] = 224
            cfg_teacher.DATA['TEMPLATE']['SIZE'] = 112
            cfg_teacher.MODEL['BACKBONE']['STRIDE'] = 14
            cfg_teacher.MODEL['HEAD']['NUM_CHANNELS'] = 192

            cfg_teacher.MODEL['BACKBONE']['TYPE'] = 'eva02_tiny_patch14_224'
            cfg_teacher['MODEL']['BACKBONE']['TYPE'] = 'eva02_tiny_patch14_224'
            net_teacher3 = build_avtrack(cfg_teacher)
            cur_path = os.path.abspath(__file__)
            pro_path = os.path.abspath(os.path.join(cur_path, '../../..'))
            checkpoint = torch.load(
                os.path.join(pro_path, 'teacher_model/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar'),
                map_location="cpu")
            checkpoint2 = torch.load(
                os.path.join(pro_path, 'teacher_model/vit_tiny_patch16_224/AVTrack_ep0300.pth.tar'),
                map_location="cpu")
            checkpoint3 = torch.load(
                os.path.join(pro_path, 'teacher_model/eva02_tiny_patch16_224/AVTrack_ep0300.pth.tar'),
                map_location="cpu")
            missing_keys, unexpected_keys = net_teacher.load_state_dict(checkpoint["net"], strict=False)
            missing_keys, unexpected_keys = net_teacher2.load_state_dict(checkpoint2["net"], strict=False)
            missing_keys, unexpected_keys = net_teacher3.load_state_dict(checkpoint3["net"], strict=False)
            net_teacher.cuda()
            net_teacher2.cuda()
            net_teacher3.cuda()
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
            cfg.MODEL['BACKBONE']['TYPE'] = 'deit_tiny_distilled_patch16_224'
            cfg['MODEL']['BACKBONE']['TYPE'] = 'deit_tiny_distilled_patch16_224'
            net = build_avtrack(cfg)
        else:
            net = build_avtrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net_teacher = DDP(net_teacher, device_ids=[settings.local_rank], find_unused_parameters=True)
        net_teacher2 = DDP(net_teacher2, device_ids=[settings.local_rank], find_unused_parameters=True)
        net_teacher3 = DDP(net_teacher3, device_ids=[settings.local_rank], find_unused_parameters=True)
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "avtrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = AVTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher
            actor.net_teacher2 = net_teacher2
            actor.net_teacher3 = net_teacher3
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    if is_distill_training:
        trainer.actor.net.is_distill_training=True
    else:
        trainer.actor.net.is_distill_training = False

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
