"""
Basic AVTrack model.
"""
import math
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.avtrack.deit import deit_tiny_patch16_224
from lib.models.avtrack.vision_transformer import vit_tiny_patch16_224
from lib.models.avtrack.eva import eva02_tiny_patch14_224
from lib.utils.box_ops import box_xyxy_to_cxcywh

import timm
from lib.utils.box_ops import box_xywh_to_xyxy
from lib.models.avtrack.loss_functions import DJSLoss
from lib.models.avtrack.statistics_network import (
    GlobalStatisticsNetwork,
)

from torch.nn.functional import l1_loss


class AVTrack(nn.Module):
    """ This is the base class for AVTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_sz_t = int(box_head.feat_sz_t)
            self.feat_len_t = int(box_head.feat_sz_t ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.djs_loss = DJSLoss()
        self.feature_map_size = 8  # 128x128
        self.feature_map_channels = transformer.embed_dim
        self.num_ch_coding = self.backbone.embed_dim
        self.coding_size = 8
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        )

        self.l1_loss = l1_loss

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_anno: torch.Tensor,
                search_anno: torch.Tensor,
                return_last_attn=False,
                ):

        if self.training:
            template_anno = torch.round(template_anno * 8).int()
            template_anno[template_anno < 0] = 0
            search_anno = torch.round(search_anno * 16).int()
            search_anno[search_anno < 0] = 0

        x, aux_dict = self.backbone(z=template, x=search,
                                    return_last_attn=return_last_attn, )

        if self.training:
            prob_active_m = torch.cat(aux_dict['probs_active'], dim=1).mean(dim=1)
            prob_active_m = prob_active_m.reshape(len(prob_active_m), 1)
            expected_active_ratio = 0.7 * torch.ones(prob_active_m.shape)
            activeness_loss = self.l1_loss(prob_active_m, expected_active_ratio.to(prob_active_m.device))
        else:
            activeness_loss = torch.zeros(0)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None, template_anno=template_anno, search_anno=search_anno)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['activeness_loss'] = activeness_loss
        return out

    def forward_head(self, cat_feature, gt_score_map=None, template_anno=None, search_anno=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if self.training:
            feat_len_t = cat_feature.shape[1] - self.feat_len_s
            feat_sz_t = int(math.sqrt(feat_len_t))
            enc_opt_z = cat_feature[:, 0:feat_len_t]
            opt = (enc_opt_z.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat_z = opt.view(-1, C, feat_sz_t, feat_sz_t)

        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        global_mutual_loss = torch.zeros(0)
        if self.training:
            opt_feat_mask = torch.zeros(cat_feature.shape[0], cat_feature.shape[2], 8, 8)
            opt_feat_x = torch.zeros(cat_feature.shape[0], cat_feature.shape[2], 8, 8)
            for i in range(opt_feat.shape[0]):
                bbox = template_anno.squeeze()[i]
                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])
                x_t = bbox[0]
                y_t = bbox[1]

                target_sz_t = opt_feat_mask[i, :, y_t:y_t + bbox[3], x_t:x_t + bbox[2]].shape

                bbox = search_anno.squeeze()[i]
                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])

                target_sz_s = opt_feat[i, :, bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]].shape
                h = min([target_sz_t[1], target_sz_s[1]])
                w = min([target_sz_t[2], target_sz_s[2]])
                opt_feat_x[i, :, y_t:y_t + h, x_t:x_t + w] = opt_feat[i, :, bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w]
                opt_feat_mask[i, :, y_t:y_t + h, x_t:x_t + w] = 1

            opt_feat_z = opt_feat_z * opt_feat_mask.to(opt_feat_z.device)

            x = opt_feat_z.to(opt_feat.device)
            y = opt_feat_x.to(opt_feat.device)
            x_shuffled = torch.cat([x[1:], x[0].unsqueeze(0)], dim=0)

            # Global mutual information estimation
            global_mutual_M_R_x = self.global_stat_x(x, y)  # positive statistic
            global_mutual_M_R_x_prime = self.global_stat_x(x_shuffled, y)
            global_mutual_loss = self.djs_loss(
                T=global_mutual_M_R_x,
                T_prime=global_mutual_M_R_x_prime,
            )

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,

                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'mine_loss': global_mutual_loss,
                   }
            return out
        else:
            raise NotImplementedError


def build_avtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('AVTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224' or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = AVTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'AVTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
