# ---------------------------------------------------------------
# vctree_head.py
# Set-up time: 2020/6/4 上午9:35
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS, builder
from mmdet.models.losses import accuracy

from .approaches import TriTransformerContext, FreqBiasModule
from .relation_head import RelationHead


@HEADS.register_module()
class TriTransformerHead(RelationHead):
    def __init__(self, **kwargs):
        super(TriTransformerHead, self).__init__(**kwargs)
        self.context_layer = TriTransformerContext(self.head_config, self.obj_classes, self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.classifier_apart = self.head_config.classifier_apart
        self.hidden_dim = self.head_config.hidden_dim
        self.num_thing_class = self.head_config.num_thing_class
        self.num_stuff_class = self.head_config.num_stuff_class
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.context_pooling_dim, self.context_pooling_dim)
        self.rel_feats_embed = nn.Linear(self.context_pooling_dim, self.embed_dim)

        self.loss_sub = builder.build_loss(self.head_config.loss_sub)
        self.loss_obj = builder.build_loss(self.head_config.loss_obj)

        self.sub_cls = nn.Linear(self.context_pooling_dim, self.num_classes, bias=True)
        self.obj_cls = nn.Linear(self.context_pooling_dim, self.num_classes, bias=True)

        if not self.classifier_apart:
            self.rel_compress = nn.Linear(self.context_pooling_dim,
                                          self.num_predicates,
                                          bias=True)
        else:
            self.thing_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.thing_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.stuff_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
            self.stuff_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)

    def init_weights(self):
        if self.with_bbox_roi_extractor:
            self.bbox_roi_extractor.init_weights()
        if self.with_relation_roi_extractor:
            self.relation_roi_extractor.init_weights()
        # self.context_layer.init_weights()
        xavier_init(self.post_emb)
        xavier_init(self.rel_feats_embed)
        xavier_init(self.sub_cls)
        xavier_init(self.obj_cls)
        if not self.classifier_apart:
            xavier_init(self.rel_compress)
        else:
            xavier_init(self.thing_stuff_classifier)
            xavier_init(self.stuff_stuff_classifier)
            xavier_init(self.thing_thing_classifier)
            xavier_init(self.stuff_thing_classifier)

    @staticmethod
    def cal_pair_accuracy(sub_preds, obj_preds, pair_labels):
        pair_accuracy = (sub_preds == pair_labels[:, 0]) & (obj_preds == pair_labels[:, 1])
        return sum(pair_accuracy) * 1.0 / pair_accuracy.shape[0]

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):

        '''roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)'''

        roi_feats, union_feats, det_result = self.prepocess_features(
            img, img_meta, det_result, gt_result
        )

        if roi_feats.shape[0] == 0:
            return det_result

        rel_feats = self.context_layer(union_feats, det_result)

        rel_feats = F.relu(self.post_emb(rel_feats))

        # split_feats = rel_feats.view(rel_feats.size(0), 2, self.hidden_dim)
        # sub_feats = split_feats[:, 0].contiguous().view(-1, self.hidden_dim)
        # obj_feats = split_feats[:, 1].contiguous().view(-1, self.hidden_dim)

        sub_dists = self.sub_cls(rel_feats)
        obj_dists = self.obj_cls(rel_feats)

        sub_pred = sub_dists[:, 1:].max(1)[1] + 1
        obj_pred = obj_dists[:, 1:].max(1)[1] + 1
        pair_pred = torch.stack([sub_pred, obj_pred], dim=-1)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        pair_dists = torch.stack([sub_dists, obj_dists], dim=1)
        pair_dists = pair_dists.split(num_rels, dim=0)

        rel_feats_embed = self.rel_feats_embed(rel_feats)

        if not self.classifier_apart:
            combine_rel_scores = self.rel_compress(rel_feats)
        else:
            split_index = torch.zeros(pair_pred.shape[0]).to(pair_pred)

            thing_thing_index = (pair_pred[:, 0] <= self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
            stuff_stuff_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] > self.num_thing_class)
            stuff_thing_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
            split_index[thing_thing_index] = 1
            split_index[stuff_stuff_index] = 2
            split_index[stuff_thing_index] = 3

            thing_stuff_split = rel_feats[split_index == 0]
            thing_thing_split = rel_feats[split_index == 1]
            stuff_stuff_split = rel_feats[split_index == 2]
            stuff_thing_split = rel_feats[split_index == 3]

            thing_stuff_rel_scores = self.thing_stuff_classifier(thing_stuff_split)
            thing_thing_rel_scores = self.thing_thing_classifier(thing_thing_split)
            stuff_stuff_rel_scores = self.stuff_stuff_classifier(stuff_stuff_split)
            stuff_thing_rel_scores = self.stuff_stuff_classifier(stuff_thing_split)

            combine_rel_scores = torch.zeros((pair_pred.shape[0], self.num_predicates)).to(sub_dists)
            combine_rel_scores[split_index == 0] = thing_stuff_rel_scores
            combine_rel_scores[split_index == 1] = thing_thing_rel_scores
            combine_rel_scores[split_index == 2] = stuff_stuff_rel_scores
            combine_rel_scores[split_index == 3] = stuff_thing_rel_scores
            # combine_rel_scores.requires_grad = True

        if self.use_bias and not self.use_penalty:
            '''rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())'''
            new_bias = None
            bias = self.freq_bias.index_with_labels(pair_pred.long())
            rel_scores = combine_rel_scores + bias
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

        refine_obj_scores = []
        for i, (rel_pair_idx, label) in enumerate(zip(det_result.rel_pair_idxes, det_result.labels)):
            single_obj_dists = torch.zeros((label.shape[0], self.num_classes))
            for j in range(label.shape[0]):
                temp_index = torch.nonzero(rel_pair_idx == j)
                single_obj_dists[j] = torch.mean(pair_dists[i][temp_index[:, 0], temp_index[:, 1], :], dim=0)
            refine_obj_scores.append(single_obj_dists)

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            target_pair_labels = []
            for i, (rel_pair_idx, target_label) in enumerate(zip(det_result.rel_pair_idxes, det_result.target_labels)):
                target_pair_labels.append(target_label[rel_pair_idx])
            det_result.target_pair_labels = torch.cat(target_pair_labels, dim=0)
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
            #refine_obj_scores = torch.cat(det_result.dists, dim=0)
        else:
            #refine_obj_scores = det_result.dists
            rel_scores = rel_scores.split(num_rels, dim=0)

        refine_obj_scores = torch.cat(refine_obj_scores, dim=0).to(sub_dists)
        det_result.refine_scores = refine_obj_scores.split(num_objs, dim=0)
        det_result.rel_scores = rel_scores
        det_result.rel_feats_embed = rel_feats_embed

        # add additional auxiliary loss
        head_spec_losses = {}

        # object-pair loss
        if not is_testing:
            if det_result.target_pair_labels is not None:
                head_spec_losses['loss_sub'] = self.loss_sub(sub_dists, det_result.target_pair_labels[:, 0]) * 0.5
                head_spec_losses['loss_obj'] = self.loss_obj(obj_dists, det_result.target_pair_labels[:, 1]) * 0.5
                valid_index = det_result.target_rel_labels > 0
                head_spec_losses['acc_object_pair'] = self.cal_pair_accuracy(sub_pred[valid_index], obj_pred[valid_index], det_result.target_pair_labels[valid_index])
                head_spec_losses['acc_object'] = accuracy(refine_obj_scores, det_result.target_labels)
            '''add_for_losses['acc_triplet'] = self.cal_triplet_accuracy(sub_pred[valid_index], obj_pred[valid_index],
                                                                       det_result.target_pair_labels[valid_index])'''
        det_result.head_spec_losses = head_spec_losses

        # ranking prediction:
        '''if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(prod_rep, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)'''

        return det_result
