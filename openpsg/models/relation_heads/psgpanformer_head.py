import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
import torchvision
import numpy as np
import mmcv
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, BitmapMasks)
from mmdet.models.utils import build_transformer
from ..panformer.detr_head_plus import RelDETRHead
from packaging import version

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


@HEADS.register_module()
class RelPanFormerHead(RelDETRHead):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            quality_threshold_things=0.25,
            quality_threshold_stuff=0.25,
            overlap_threshold_things=0.4,
            overlap_threshold_stuff=0.2,
            use_argmax=False,
            datasets='coco',  # MDS
            sub_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            obj_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            dataset_config=None,
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = quality_threshold_things
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.use_argmax = use_argmax
        self.datasets = datasets
        self.fp16_enabled = False

        # MDS: id_and_category_maps is the category_dict
        if datasets == 'coco':
            from ..panformer.color_map import id_and_category_maps
            self.cat_dict = id_and_category_maps
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_sub = sub_transformer_head['num_decoder_layers']
        self.num_dec_obj = obj_transformer_head['num_decoder_layers']
        super(RelPanFormerHead, self).__init__(*args,
                                            transformer=transformer,
                                            train_cfg=train_cfg,
                                            **kwargs)

        # self.loss_mask = build_loss(loss_mask)
        self.sub_mask_head = build_transformer(sub_transformer_head)
        self.obj_mask_head = build_transformer(obj_transformer_head)
        self.count = 0

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        obj_fc_cls = Linear(self.embed_dims, self.obj_cls_out_channels)
        rel_fc_cls = Linear(self.embed_dims, self.rel_cls_out_channels)
        # fc_cls_stuff = Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.s_cls_branches = _get_clones(obj_fc_cls, num_pred)
            self.o_cls_branches = _get_clones(obj_fc_cls, num_pred)
            self.r_cls_branches = _get_clones(rel_fc_cls, num_pred)
            self.s_reg_branches = _get_clones(reg_branch, num_pred)
            self.o_reg_branches = _get_clones(reg_branch, num_pred)
            self.r_reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.s_cls_branches = nn.ModuleList(
                [obj_fc_cls for _ in range(num_pred)])
            self.o_cls_branches = nn.ModuleList(
                [obj_fc_cls for _ in range(num_pred)])
            self.r_cls_branches = nn.ModuleList(
                [rel_fc_cls for _ in range(num_pred)])
            self.s_reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.o_reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.r_reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        #self.stuff_query = nn.Embedding(self.num_stuff_classes,
        #                               self.embed_dims * 2)
        #self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)  # used in mask decoder
        #self.cls_thing_branches = _get_clones(obj_fc_cls, self.num_dec_things)  # used in mask decoder
        #self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # used in mask deocder

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.s_loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.s_cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.o_cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.r_cls_branches:
                nn.init.constant_(m.bias, bias_init)
            '''for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)'''
        for m in self.s_reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.o_reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.r_reg_branches:
            constant_init(m[-1], 0, bias=0)
        '''for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)'''
        nn.init.constant_(self.s_reg_branches[0][-1].bias.data[2:], -2.0)
        nn.init.constant_(self.o_reg_branches[0][-1].bias.data[2:], -2.0)
        nn.init.constant_(self.r_reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.s_reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            for m in self.o_reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
            '''for m in self.r_reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)'''

    def _get_target_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_preds,
                           o_mask_preds,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            s_cls_score (Tensor): Subject box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            o_cls_score (Tensor): Object box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            r_cls_score (Tensor): Relation score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            s_bbox_pred (Tensor): Sigmoid outputs of Subject bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            o_bbox_pred (Tensor): Sigmoid outputs of object bboxes from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            s_mask_preds (Tensor): Logits before sigmoid subject masks from a single decoder layer
                for one image, with shape [num_query, H, W].
            o_mask_preds (Tensor): Logits before sigmoid object masks from a single decoder layer
                for one image, with shape [num_query, H, W].
            gt_rels (Tensor): Ground truth relation triplets for one image with
                shape (num_gts, 3) in [gt_sub_id, gt_obj_id, gt_rel_class] format.
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - s/o/r_labels (Tensor): Labels of each image.
                - s/o/r_label_weights (Tensor]): Label weights of each image.
                - s/o_bbox_targets (Tensor): BBox targets of each image.
                - s/o_bbox_weights (Tensor): BBox weights of each image.
                - s/o_mask_targets (Tensor): Mask targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
                - s/o_mask_preds (Tensor): Matched mask preds of each image.
        """

        num_bboxes = s_bbox_pred.size(0)
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        if self.use_mask:
            gt_sub_masks = []
            gt_obj_masks = []

        assert len(gt_masks) == len(gt_bboxes)

        for rel_id in range(gt_rels.size(0)):
            gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            if self.use_mask:
                gt_sub_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         0])].unsqueeze(0))
                gt_obj_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         1])].unsqueeze(0))

        gt_sub_bboxes = torch.vstack(gt_sub_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_obj_bboxes = torch.vstack(gt_obj_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_sub_labels = torch.vstack(gt_sub_labels).type_as(gt_labels).reshape(
            -1)
        gt_obj_labels = torch.vstack(gt_obj_labels).type_as(gt_labels).reshape(
            -1)
        gt_rel_labels = torch.vstack(gt_rel_labels).type_as(gt_labels).reshape(
            -1)

        # assigner and sampler, only return subject&object assign result
        s_assign_result, o_assign_result = self.assigner.assign(
            s_bbox_pred, o_bbox_pred, s_cls_score, o_cls_score, r_cls_score,
            gt_sub_bboxes, gt_obj_bboxes, gt_sub_labels, gt_obj_labels,
            gt_rel_labels, img_meta, gt_bboxes_ignore)

        s_sampling_result = self.sampler.sample(s_assign_result, s_bbox_pred,
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, o_bbox_pred,
                                                gt_obj_bboxes)
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  #### no-rel class indices in prediction

        # label targets
        s_labels = gt_sub_bboxes.new_full(
            (num_bboxes,), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes]  as background
        s_labels[pos_inds] = gt_sub_labels[
            s_sampling_result.pos_assigned_gt_inds]
        s_label_weights = gt_sub_bboxes.new_zeros(num_bboxes)
        s_label_weights[pos_inds] = 1.0  # setting the weight of positive samples to 1.0

        o_labels = gt_obj_bboxes.new_full(
            (num_bboxes,), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes] as background
        o_labels[pos_inds] = gt_obj_labels[
            o_sampling_result.pos_assigned_gt_inds]
        o_label_weights = gt_obj_bboxes.new_zeros(num_bboxes)
        o_label_weights[pos_inds] = 1.0

        r_labels = gt_obj_bboxes.new_full(
            (num_bboxes,), 0,
            dtype=torch.long)  ### 1-based, class 0 as background
        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_bboxes)

        if self.use_mask:

            gt_sub_masks = torch.cat(gt_sub_masks, axis=0).type_as(gt_masks[0])
            gt_obj_masks = torch.cat(gt_obj_masks, axis=0).type_as(gt_masks[0])

            assert gt_sub_masks.size() == gt_obj_masks.size()
            # mask targets for subjects and objects
            s_mask_targets = gt_sub_masks[
                s_sampling_result.pos_assigned_gt_inds,
                ...]
            s_mask_preds = s_mask_preds[pos_inds]

            o_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = o_mask_preds[pos_inds]

            s_mask_preds = interpolate(s_mask_preds[:, None],
                                       size=gt_sub_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)

            o_mask_preds = interpolate(o_mask_preds[:, None],
                                       size=gt_obj_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)
        else:
            s_mask_targets = None
            s_mask_preds = None
            o_mask_targets = None
            o_mask_preds = None

        # bbox targets for subjects and objects
        s_bbox_targets = torch.zeros_like(s_bbox_pred)
        s_bbox_weights = torch.zeros_like(s_bbox_pred)
        s_bbox_weights[pos_inds] = 1.0

        o_bbox_targets = torch.zeros_like(o_bbox_pred)
        o_bbox_weights = torch.zeros_like(o_bbox_pred)
        o_bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = o_bbox_pred.new_tensor([img_w, img_h, img_w,
                                         img_h]).unsqueeze(0)

        pos_gt_s_bboxes_normalized = s_sampling_result.pos_gt_bboxes / factor
        pos_gt_s_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_s_bboxes_normalized)
        s_bbox_targets[pos_inds] = pos_gt_s_bboxes_targets

        pos_gt_o_bboxes_normalized = o_sampling_result.pos_gt_bboxes / factor
        pos_gt_o_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_o_bboxes_normalized)
        o_bbox_targets[pos_inds] = pos_gt_o_bboxes_targets

        return (s_labels, o_labels, r_labels, s_label_weights, o_label_weights,
                r_label_weights, s_bbox_targets, o_bbox_targets,
                s_bbox_weights, o_bbox_weights, s_mask_targets, o_mask_targets,
                pos_inds, neg_inds, s_mask_preds, o_mask_preds
                )  ###return the interpolated predicted masks

    def get_targets(self,
                    s_cls_scores_list,
                    o_cls_scores_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    s_mask_preds_list,
                    o_mask_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(s_cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (s_labels_list, o_labels_list, r_labels_list, s_label_weights_list,
         o_label_weights_list, r_label_weights_list, s_bbox_targets_list,
         o_bbox_targets_list, s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list, pos_inds_list,
         neg_inds_list, s_mask_preds_list, o_mask_preds_list) = multi_apply(
             self._get_target_single, s_cls_scores_list, o_cls_scores_list,
             r_cls_scores_list, s_bbox_preds_list, o_bbox_preds_list,
             s_mask_preds_list, o_mask_preds_list, gt_rels_list,
             gt_bboxes_list, gt_labels_list, gt_masks_list, img_metas,
             gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (s_labels_list, o_labels_list, r_labels_list,
                s_label_weights_list, o_label_weights_list,
                r_label_weights_list, s_bbox_targets_list, o_bbox_targets_list,
                s_bbox_weights_list, o_bbox_weights_list, s_mask_targets_list,
                o_mask_targets_list, num_total_pos, num_total_neg,
                s_mask_preds_list, o_mask_preds_list)

    def loss_single(self,
                    s_cls_scores,
                    o_cls_scores,
                    r_cls_scores,
                    s_bbox_preds,
                    o_bbox_preds,
                    s_mask_preds,
                    o_mask_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        num_imgs = s_cls_scores.size(0)

        s_cls_scores_list = [s_cls_scores[i] for i in range(num_imgs)]
        o_cls_scores_list = [o_cls_scores[i] for i in range(num_imgs)]
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        if self.use_mask:
            s_mask_preds_list = [s_mask_preds[i] for i in range(num_imgs)]
            o_mask_preds_list = [o_mask_preds[i] for i in range(num_imgs)]
        else:
            s_mask_preds_list = [None for i in range(num_imgs)]
            o_mask_preds_list = [None for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            s_cls_scores_list, o_cls_scores_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list, s_mask_preds_list,
            o_mask_preds_list, gt_rels_list, gt_bboxes_list, gt_labels_list,
            gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (s_labels_list, o_labels_list, r_labels_list, s_label_weights_list,
         o_label_weights_list, r_label_weights_list, s_bbox_targets_list,
         o_bbox_targets_list, s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list, num_total_pos,
         num_total_neg, s_mask_preds_list, o_mask_preds_list) = cls_reg_targets
        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)

        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        s_bbox_targets = torch.cat(s_bbox_targets_list, 0)
        o_bbox_targets = torch.cat(o_bbox_targets_list, 0)

        s_bbox_weights = torch.cat(s_bbox_weights_list, 0)
        o_bbox_weights = torch.cat(o_bbox_weights_list, 0)

        if self.use_mask:
            s_mask_targets = torch.cat(s_mask_targets_list,
                                       0).float().flatten(1)
            o_mask_targets = torch.cat(o_mask_targets_list,
                                       0).float().flatten(1)

            s_mask_preds = torch.cat(s_mask_preds_list, 0).flatten(1)
            o_mask_preds = torch.cat(o_mask_preds_list, 0).flatten(1)
            num_matches = o_mask_preds.shape[0]

            # mask loss
            # s_focal_loss = self.sub_focal_loss(s_mask_preds,s_mask_targets,num_matches)
            s_dice_loss = self.s_dice_loss(
                s_mask_preds, s_mask_targets,
                num_matches)

            # o_focal_loss = self.obj_focal_loss(o_mask_preds,o_mask_targets,num_matches)
            o_dice_loss = self.o_dice_loss(
                o_mask_preds, o_mask_targets,
                num_matches)
        else:
            s_dice_loss = None
            o_dice_loss = None

        # classification loss
        s_cls_scores = s_cls_scores.reshape(-1, self.obj_cls_out_channels)
        o_cls_scores = o_cls_scores.reshape(-1, self.obj_cls_out_channels)
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                s_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        ###NOTE change cls_avg_factor for objects as we do not calculate object classification loss for unmatched ones

        s_loss_cls = self.s_loss_cls(s_cls_scores,
                                       s_labels,
                                       s_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        o_loss_cls = self.o_loss_cls(o_cls_scores,
                                       o_labels,
                                       o_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        r_loss_cls = self.r_loss_cls(r_cls_scores,
                                       r_labels,
                                       r_label_weights,
                                       avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = o_loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, s_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        s_bbox_preds = s_bbox_preds.reshape(-1, 4)
        s_bboxes = bbox_cxcywh_to_xyxy(s_bbox_preds) * factors
        s_bboxes_gt = bbox_cxcywh_to_xyxy(s_bbox_targets) * factors

        o_bbox_preds = o_bbox_preds.reshape(-1, 4)
        o_bboxes = bbox_cxcywh_to_xyxy(o_bbox_preds) * factors
        o_bboxes_gt = bbox_cxcywh_to_xyxy(o_bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        s_loss_iou = self.s_loss_iou(s_bboxes,
                                       s_bboxes_gt,
                                       s_bbox_weights,
                                       avg_factor=num_total_pos)
        o_loss_iou = self.o_loss_iou(o_bboxes,
                                       o_bboxes_gt,
                                       o_bbox_weights,
                                       avg_factor=num_total_pos)

        # regression L1 loss
        s_loss_bbox = self.s_loss_bbox(s_bbox_preds,
                                         s_bbox_targets,
                                         s_bbox_weights,
                                         avg_factor=num_total_pos)
        o_loss_bbox = self.o_loss_bbox(o_bbox_preds,
                                         o_bbox_targets,
                                         o_bbox_weights,
                                         avg_factor=num_total_pos)
        # return s_loss_cls, o_loss_cls, r_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_focal_loss, s_dice_loss, o_focal_loss, o_dice_loss
        return s_loss_cls, o_loss_cls, r_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_dice_loss, o_dice_loss

    @force_fp32(apply_to=('mlvl_feats',))
    def forward(self, mlvl_feats, img_metas=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)

        input_img_h, input_img_w = img_metas[0]['batch_input_shape']

        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.r_reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.r_cls_branches if self.as_two_stage else None  # noqa:E501
        )

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        len_last_feat = hw_lvl[-1][0] * hw_lvl[-1][1]

        # we should feed these to mask deocder.
        memory, memory_mask, memory_pos = memory[:, :-len_last_feat, :], \
                      memory_mask[:, :-len_last_feat], \
                      memory_pos[:, :-len_last_feat, :]

        hs = hs.permute(0, 2, 1, 3)
        r_outputs_classes, s_outputs_classes, o_outputs_classes = [], [], []
        r_outputs_coords, s_outputs_coords, o_outputs_coords = [], [], []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            r_outputs_class = self.r_cls_branches[lvl](hs[lvl])
            s_outputs_class = self.s_cls_branches[lvl](hs[lvl])
            o_outputs_class = self.o_cls_branches[lvl](hs[lvl])
            # r_tmp = self.r_reg_branches[lvl](hs[lvl])
            s_tmp = self.s_reg_branches[lvl](hs[lvl])
            o_tmp = self.o_reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                # r_tmp += reference
                s_tmp += reference
                o_tmp += reference
            else:
                assert reference.shape[-1] == 2
                # r_tmp[..., :2] += reference
                s_tmp[..., :2] += reference
                o_tmp[..., :2] += reference
            # r_outputs_coord = r_tmp.sigmoid()
            s_outputs_coord = s_tmp.sigmoid()
            o_outputs_coord = o_tmp.sigmoid()
            r_outputs_classes.append(r_outputs_class)
            s_outputs_classes.append(s_outputs_class)
            o_outputs_classes.append(o_outputs_class)
            # r_outputs_coords.append(r_outputs_coord)
            s_outputs_coords.append(s_outputs_coord)
            o_outputs_coords.append(o_outputs_coord)

        r_outputs_classes = torch.stack(r_outputs_classes)
        s_outputs_classes = torch.stack(s_outputs_classes)
        o_outputs_classes = torch.stack(o_outputs_classes)
        # r_outputs_coords = torch.stack(r_outputs_coords)
        s_outputs_coords = torch.stack(s_outputs_coords)
        o_outputs_coords = torch.stack(o_outputs_coords)

        if self.use_mask:
            sub_mask, sub_mask_inter, sub_query_inter = self.sub_mask_head(
                memory, memory_mask, None, query, None, None, hw_lvl=hw_lvl
            )
            obj_mask, obj_mask_inter, obj_query_inter = self.obj_mask_head(
                memory, memory_mask, None, query, None, None, hw_lvl=hw_lvl
            )

            sub_mask = sub_mask.squeeze(-1).reshape(batch_size, self.num_query, *hw_lvl[0])
            obj_mask = obj_mask.squeeze(-1).reshape(batch_size, self.num_query, *hw_lvl[0])

        all_bbox_preds = dict(sub=s_outputs_coords,
                              obj=o_outputs_coords,
                              sub_seg=sub_mask,
                              obj_seg=obj_mask)
        all_cls_scores = dict(sub=s_outputs_classes,
                              obj=o_outputs_classes,
                              rel=r_outputs_classes)

        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(
            self,
            all_cls_scores_list,
            all_bbox_preds_list,
            gt_rels_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list=None,
            img_metas=None,
            gt_bboxes_ignore=None,
    ):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            args_tuple (Tuple) several args
            reference (Tensor) reference from location decoder
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        all_s_cls_scores = all_cls_scores['sub']
        all_o_cls_scores = all_cls_scores['obj']

        all_s_bbox_preds = all_bbox_preds['sub']
        all_o_bbox_preds = all_bbox_preds['obj']

        num_dec_layers = len(all_s_cls_scores)

        if self.use_mask:
            all_s_mask_preds = all_bbox_preds['sub_seg']
            all_o_mask_preds = all_bbox_preds['obj_seg']
            all_s_mask_preds = [
                all_s_mask_preds for _ in range(num_dec_layers)
            ]
            all_o_mask_preds = [
                all_o_mask_preds for _ in range(num_dec_layers)
            ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = [None for _ in range(num_dec_layers)]
        all_r_cls_scores = all_cls_scores['rel']

        if self.use_mask:
            # s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_focal_losses, s_dice_losses, o_focal_losses, o_dice_losses = multi_apply(
            #     self.loss_single, all_s_cls_scores, all_o_cls_scores, all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
            #     all_s_mask_preds, all_o_mask_preds,
            #     all_gt_rels_list,all_gt_bboxes_list, all_gt_labels_list,
            #     all_gt_masks_list, img_metas_list,
            #     all_gt_bboxes_ignore_list)
            s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses = multi_apply(
                self.loss_single, all_s_cls_scores, all_o_cls_scores,
                all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
                all_s_mask_preds, all_o_mask_preds, all_gt_rels_list,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
                img_metas_list, all_gt_bboxes_ignore_list)
        else:
            all_s_mask_preds = [None for _ in range(num_dec_layers)]
            all_o_mask_preds = [None for _ in range(num_dec_layers)]
            s_losses_cls, o_losses_cls, r_losses_cls, s_losses_bbox, o_losses_bbox, s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses = multi_apply(
                self.loss_single, all_s_cls_scores, all_o_cls_scores,
                all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
                all_s_mask_preds, all_o_mask_preds, all_gt_rels_list,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list,
                img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['s_loss_cls'] = s_losses_cls[-1]
        loss_dict['o_loss_cls'] = o_losses_cls[-1]
        loss_dict['r_loss_cls'] = r_losses_cls[-1]
        loss_dict['s_loss_bbox'] = s_losses_bbox[-1]
        loss_dict['o_loss_bbox'] = o_losses_bbox[-1]
        loss_dict['s_loss_iou'] = s_losses_iou[-1]
        loss_dict['o_loss_iou'] = o_losses_iou[-1]
        if self.use_mask:
            # loss_dict['s_focal_losses'] = s_focal_losses[-1]
            # loss_dict['o_focal_losses'] = o_focal_losses[-1]
            loss_dict['s_dice_losses'] = s_dice_losses[-1]
            loss_dict['o_dice_losses'] = o_dice_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                          s_losses_bbox[:-1], o_losses_bbox[:-1],
                                          s_losses_iou[:-1], o_losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('x',))
    def forward_train(self,
                      x,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_semantic_seg=None,
                      **kwargs):
        """
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, img_metas)
        else:
            if gt_masks is None:
                loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, img_metas)
            else:
                loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks,
                                      img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def _get_bboxes_single(self,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_pred,
                           o_mask_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        assert len(s_cls_score) == len(o_cls_score)
        assert len(s_cls_score) == len(s_bbox_pred)
        assert len(s_cls_score) == len(o_bbox_pred)

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        assert self.s_loss_cls.use_sigmoid == False
        assert self.o_loss_cls.use_sigmoid == False
        assert self.r_loss_cls.use_sigmoid == False
        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index] + 1
        s_bbox_pred = s_bbox_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index] + 1
        o_bbox_pred = o_bbox_pred[triplet_index]

        r_dists = r_lgs.reshape(
            -1, self.num_relations +
                1)[triplet_index]  #### NOTE: to match the evaluation in vg

        if self.use_mask:
            s_mask_pred = s_mask_pred[triplet_index]
            o_mask_pred = o_mask_pred[triplet_index]
            s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)

            s_mask_pred_logits = s_mask_pred
            o_mask_pred_logits = o_mask_pred

            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
            ### triplets deduplicate####
            relation_classes = defaultdict(lambda: [])
            for k, (s_l, o_l, r_l) in enumerate(zip(s_labels, o_labels, r_labels)):
                relation_classes[(s_l.item(), o_l.item(), r_l.item())].append(k)
            s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
            o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

            def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
                while len(triplets_ids) > 1:
                    base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
                    base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
                    other_s_masks = s_binary_masks[triplets_ids[1:]]
                    other_o_masks = o_binary_masks[triplets_ids[1:]]
                    # calculate ious
                    s_ious = base_s_mask.mm(other_s_masks.transpose(0, 1)) / ((base_s_mask + other_s_masks) > 0).sum(-1)
                    o_ious = base_o_mask.mm(other_o_masks.transpose(0, 1)) / ((base_o_mask + other_o_masks) > 0).sum(-1)
                    ids_left = []
                    for s_iou, o_iou, other_id in zip(s_ious[0], o_ious[0], triplets_ids[1:]):
                        if (s_iou > 0.5) & (o_iou > 0.5):
                            keep_tri[other_id] = False
                        else:
                            ids_left.append(other_id)
                    triplets_ids = ids_left
                return keep_tri

            keep_tri = torch.ones_like(r_labels, dtype=torch.bool)
            for triplets_ids in relation_classes.values():
                if len(triplets_ids) > 1:
                    keep_tri = dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

            s_labels = s_labels[keep_tri]
            o_labels = o_labels[keep_tri]
            s_mask_pred = s_mask_pred[keep_tri]
            o_mask_pred = o_mask_pred[keep_tri]

            complete_labels = torch.cat((s_labels, o_labels), 0)
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)
            r_scores = r_scores[keep_tri]
            r_labels = r_labels[keep_tri]
            r_dists = r_dists[keep_tri]
            rel_pairs = torch.arange(keep_tri.sum() * 2,
                                     dtype=torch.int).reshape(2, -1).T
            complete_r_labels = r_labels
            complete_r_dists = r_dists

            s_binary_masks = s_binary_masks[keep_tri]
            o_binary_masks = o_binary_masks[keep_tri]

            s_mask_pred_logits = s_mask_pred_logits[keep_tri]
            o_mask_pred_logits = o_mask_pred_logits[keep_tri]

            ###end triplets deduplicate####

            #### for panoptic postprocessing ####
            keep = (s_labels != (s_logits.shape[-1] - 1)) & (
                    o_labels != (s_logits.shape[-1] - 1)) & (
                           s_scores[keep_tri] > 0.5) & (o_scores[keep_tri] > 0.5) & (
                               r_scores > 0.3)  ## the threshold is set to 0.85
            r_scores = r_scores[keep]
            r_labels = r_labels[keep]
            r_dists = r_dists[keep]

            labels = torch.cat((s_labels[keep], o_labels[keep]), 0) - 1
            masks = torch.cat((s_mask_pred[keep], o_mask_pred[keep]), 0)
            binary_masks = masks.to(torch.float).flatten(1)
            s_mask_pred_logits = s_mask_pred_logits[keep]
            o_mask_pred_logits = o_mask_pred_logits[keep]
            mask_logits = torch.cat((s_mask_pred_logits, o_mask_pred_logits), 0)

            h, w = masks.shape[-2:]

            if labels.numel() == 0:
                pan_img = torch.ones(mask_size).cpu().to(torch.long)
                pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(masks.device).reshape(2, -1).T
                rels = torch.tensor([0, 0, 0]).view(-1, 3)
                pan_labels = torch.tensor([0])
            else:
                stuff_equiv_classes = defaultdict(lambda: [])
                thing_classes = defaultdict(lambda: [])
                thing_dedup = defaultdict(lambda: [])
                for k, label in enumerate(labels):
                    if label.item() >= 80:
                        stuff_equiv_classes[label.item()].append(k)
                    else:
                        thing_classes[label.item()].append(k)

                pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(masks.device)

                def dedup_things(pred_ids, binary_masks):
                    while len(pred_ids) > 1:
                        base_mask = binary_masks[pred_ids[0]].unsqueeze(0)
                        other_masks = binary_masks[pred_ids[1:]]
                        # calculate ious
                        ious = base_mask.mm(other_masks.transpose(0, 1)) / ((base_mask + other_masks) > 0).sum(-1)
                        ids_left = []
                        thing_dedup[pred_ids[0]].append(pred_ids[0])
                        for iou, other_id in zip(ious[0], pred_ids[1:]):
                            if iou > 0.5:
                                thing_dedup[pred_ids[0]].append(other_id)
                            else:
                                ids_left.append(other_id)
                        pred_ids = ids_left
                    if len(pred_ids) == 1:
                        thing_dedup[pred_ids[0]].append(pred_ids[0])

                # create dict that groups duplicate masks
                for thing_pred_ids in thing_classes.values():
                    if len(thing_pred_ids) > 1:
                        dedup_things(thing_pred_ids, binary_masks)
                    else:
                        thing_dedup[thing_pred_ids[0]].append(thing_pred_ids[0])

                def get_ids_area(masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                    # This helper function creates the final panoptic segmentation image
                    # It also returns the area of the masks that appears on the image
                    masks = masks.flatten(1)
                    m_id = masks.transpose(0, 1).softmax(-1)

                    if m_id.shape[-1] == 0:
                        # We didn't detect any mask :(
                        m_id = torch.zeros((h, w),
                                           dtype=torch.long,
                                           device=m_id.device)
                    else:
                        m_id = m_id.argmax(-1).view(h, w)

                    if dedup:
                        # Merge the masks corresponding to the same stuff class
                        for equiv in stuff_equiv_classes.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]
                        # Merge the masks corresponding to the same thing instance
                        for equiv in thing_dedup.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs[eq_id] = equiv[0]

                    m_ids_remain, _ = m_id.unique().sort()

                    pan_rel_pairs = pan_rel_pairs.reshape(2, -1).T
                    no_obj_filter = torch.zeros(pan_rel_pairs.shape[0], dtype=torch.bool)
                    for triplet_id in range(pan_rel_pairs.shape[0]):
                        if pan_rel_pairs[triplet_id, 0] in m_ids_remain and pan_rel_pairs[
                            triplet_id, 1] in m_ids_remain:
                            no_obj_filter[triplet_id] = True
                    pan_rel_pairs = pan_rel_pairs[no_obj_filter]
                    r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]
                    pan_labels = []
                    pan_masks = []
                    for i, m_id_remain in enumerate(m_ids_remain):
                        pan_masks.append(m_id.eq(m_id_remain).unsqueeze(0))
                        pan_labels.append(labels[m_id_remain].unsqueeze(0))
                        m_id.masked_fill_(m_id.eq(m_id_remain), i)
                        pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(m_id_remain), i)
                    pan_masks = torch.cat(pan_masks, 0)
                    pan_labels = torch.cat(pan_labels, 0)
                    seg_img = m_id * INSTANCE_OFFSET + pan_labels[m_id]
                    seg_img = seg_img.view(h, w).cpu().to(torch.long)
                    m_id = m_id.view(h, w).cpu()
                    area = []
                    for i in range(len(masks)):
                        area.append(m_id.eq(i).sum().item())
                    return area, seg_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels

                area, pan_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels = get_ids_area(mask_logits,
                                                                                                      pan_rel_pairs,
                                                                                                      r_labels, r_dists,
                                                                                                      dedup=True)
                if r_labels.numel() == 0:
                    rels = torch.tensor([0, 0, 0]).view(-1, 3)
                else:
                    rels = torch.cat((pan_rel_pairs, r_labels.unsqueeze(-1)), -1)
                # if labels.numel() > 0:
                #     # We know filter empty masks as long as we find some
                #     while True:
                #         filtered_small = torch.as_tensor(
                #             [area[i] <= 4 for i, c in enumerate(labels)],
                #             dtype=torch.bool,
                #             device=keep.device)
                #         if filtered_small.any().item():
                #             scores = scores[~filtered_small]
                #             labels = labels[~filtered_small]
                #             masks = masks[~filtered_small]
                #             area, pan_img = get_ids_area(masks, scores)
                #         else:
                #             break

        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        det_bboxes = torch.cat((s_det_bboxes[keep_tri], o_det_bboxes[keep_tri]), 0)

        if self.use_mask:
            return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                   pan_img, complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks, rels, pan_labels
        else:
            return det_bboxes, labels, rel_pairs, r_labels, r_dists

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):

        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.

        result_list = []
        for img_id in range(len(img_metas)):
            s_cls_score = cls_scores['sub'][-1, img_id, ...]
            o_cls_score = cls_scores['obj'][-1, img_id, ...]
            r_cls_score = cls_scores['rel'][-1, img_id, ...]
            s_bbox_pred = bbox_preds['sub'][-1, img_id, ...]
            o_bbox_pred = bbox_preds['obj'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if self.use_mask:
                s_mask_pred = bbox_preds['sub_seg'][img_id, ...]
                o_mask_pred = bbox_preds['obj_seg'][img_id, ...]
            else:
                s_mask_pred = None
                o_mask_pred = None
            triplets = self._get_bboxes_single(s_cls_score, o_cls_score,
                                               r_cls_score, s_bbox_pred,
                                               o_bbox_pred, s_mask_pred,
                                               o_mask_pred, img_shape,
                                               scale_factor, rescale)
            result_list.append(triplets)

        return result_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False):

        # forward of this head requires img_metas
        # start = time.time()
        outs = self.forward(feats, img_metas)
        # forward_time =time.time()
        # print('------forward-----')
        # print(forward_time - start)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        # print('-----get_bboxes-----')
        # print(time.time() - forward_time)
        return results_list


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
    """Equivalent to nn.functional.interpolate, but with support for empty
    batch sizes.

    This will eventually be supported natively by PyTorch, and this class can
    go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor,
                                                   mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor,
                                                mode, align_corners)