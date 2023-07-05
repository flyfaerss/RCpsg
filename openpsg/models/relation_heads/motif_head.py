# ---------------------------------------------------------------
# motif_head.py
# Set-up time: 2020/4/27 下午8:08
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS

from .approaches import LSTMContext
from .relation_head import RelationHead


@HEADS.register_module()
class MotifHead(RelationHead):
    def __init__(self, **kwargs):
        super(MotifHead, self).__init__(**kwargs)

        self.context_layer = LSTMContext(self.head_config, self.obj_classes,
                                         self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2,
                                  self.context_pooling_dim)
        self.rel_compress = nn.Linear(self.context_pooling_dim,
                                      self.num_predicates,
                                      bias=True)
        self.rel_feats_embed = nn.Linear(self.context_pooling_dim, self.embed_dim)

        self.classifier_apart = self.head_config.classifier_apart

        if not self.classifier_apart:
            self.rel_compress = nn.Linear(self.context_pooling_dim,
                                          self.num_predicates,
                                          bias=True)
        else:
            self.thing_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.thing_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.stuff_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.stuff_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)

        if self.head_config.feature_extract_method == 'roi' and self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim,
                                    self.context_pooling_dim)
        else:
            self.union_single_not_match = False

    def init_weights(self):
        if self.with_bbox_roi_extractor:
            self.bbox_roi_extractor.init_weights()
        if self.with_relation_roi_extractor:
            self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        normal_init(self.post_emb,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim)**0.5)
        xavier_init(self.post_cat)
        xavier_init(self.rel_feats_embed)
        if not self.classifier_apart:
            xavier_init(self.rel_compress)
        else:
            xavier_init(self.thing_stuff_classifier)
            xavier_init(self.stuff_stuff_classifier)
            xavier_init(self.thing_thing_classifier)
            xavier_init(self.stuff_thing_classifier)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def forward(
        self,
        img,
        img_meta,
        det_result,
        gt_result=None,
        is_testing=False,
        ignore_classes=None,
    ):
        """
        Obtain the relation prediction results based on detection results.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:

        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        '''roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)'''

        roi_feats, union_feats, det_result = self.prepocess_features(
            img, img_meta, det_result, gt_result
        )

        if roi_feats.shape[0] == 0:
            return det_result

        # (N_b, N_c + 1), (N_b),
        refine_obj_scores, obj_preds, edge_ctx, _ = self.context_layer(
            roi_feats, det_result)

        if is_testing and ignore_classes is not None:
            refine_obj_scores = self.process_ignore_objects(
                refine_obj_scores, ignore_classes)
            obj_preds = refine_obj_scores[:, 1:].max(1)[1] + 1

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_feats)
            else:
                prod_rep = prod_rep * union_feats

        rel_feats_embed = self.rel_feats_embed(prod_rep)
        # rel_scores = self.rel_compress(prod_rep)

        if not self.classifier_apart:
            combine_rel_scores = self.rel_compress(prod_rep)
        else:
            split_index = torch.zeros(pair_pred.shape[0]).to(pair_pred)

            thing_thing_index = (pair_pred[:, 0] <= self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
            stuff_stuff_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] > self.num_thing_class)
            stuff_thing_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
            split_index[thing_thing_index] = 1
            split_index[stuff_stuff_index] = 2
            split_index[stuff_thing_index] = 3

            thing_stuff_split = prod_rep[split_index == 0]
            thing_thing_split = prod_rep[split_index == 1]
            stuff_stuff_split = prod_rep[split_index == 2]
            stuff_thing_split = prod_rep[split_index == 3]

            thing_stuff_rel_scores = self.thing_stuff_classifier(thing_stuff_split)
            thing_thing_rel_scores = self.thing_thing_classifier(thing_thing_split)
            stuff_stuff_rel_scores = self.stuff_stuff_classifier(stuff_stuff_split)
            stuff_thing_rel_scores = self.stuff_stuff_classifier(stuff_thing_split)

            combine_rel_scores = torch.zeros((pair_pred.shape[0], self.num_predicates)).to(refine_obj_scores)
            combine_rel_scores[split_index == 0] = thing_stuff_rel_scores
            combine_rel_scores[split_index == 1] = thing_thing_rel_scores
            combine_rel_scores[split_index == 2] = stuff_stuff_rel_scores
            combine_rel_scores[split_index == 3] = stuff_thing_rel_scores
        # combine_rel_scores.requires_grad = True

        '''if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())'''
        new_bias = None
        resistance_bias = None
        bias = None
        rel_scores = combine_rel_scores
        det_result.init_rel_scores = None
        if self.use_bias and not self.use_penalty:
            '''rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())'''
            bias = self.freq_bias.index_with_labels(pair_pred.long())
            rel_scores = combine_rel_scores + bias
            '''if self.training:
                rel_scores[:, 0] += np.log(0.02)'''
        elif self.use_penalty:
            new_bias, bias, resistance_bias = self.bias_module.index_with_labels(
                pair_pred.long(), epoch=det_result.epoch, max_epochs=det_result.max_epochs)
            rel_scores = combine_rel_scores + new_bias
            if bias is not None:
                det_result.init_rel_scores = combine_rel_scores + bias
            else:
                det_result.init_rel_scores = combine_rel_scores

        det_result.new_bias = new_bias
        det_result.bias = bias
        det_result.resistance_bias = resistance_bias

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = (torch.cat(
                det_result.target_rel_labels,
                dim=-1) if det_result.target_rel_labels is not None else None)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        det_result.rel_feats_embed = rel_feats_embed

        # ranking prediction:
        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(prod_rep, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)

        return det_result
