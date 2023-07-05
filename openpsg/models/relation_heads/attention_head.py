import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS

from .approaches import RelAttnContext
from .relation_head import RelationHead
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class RelAttnHead(RelationHead):
    def __init__(self, **kwargs):
        super(RelAttnHead, self).__init__(**kwargs)

        self.context_layer = RelAttnContext(
            self.head_config,
        )
        self.context_layer = build_transformer(self.transformer)

        self.use_obj_refine = self.head_config.use_obj_refine
        self.hidden_dim = self.head_config.hidden_dim
        self.rel_query_dim = self.head_config.rel_query_dim

        # post classification
        self.rel_classifier = nn.Linear(self.rel_query_dim, self.num_predicates)
        self.sub_classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.obj_classifier = nn.Linear(self.hidden_dim, self.num_classes)

        self.embed_dim = self.head_config.embed_dim
        self.rel_feats_embed = nn.Linear(self.hidden_dim, self.embed_dim)

        self.thing_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
        self.thing_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
        self.stuff_stuff_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
        self.stuff_thing_classifier = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)

    def init_weights(self):
        self.context_layer.init_weights()
        xavier_init(self.rel_classifier)
        xavier_init(self.sub_classifier)
        xavier_init(self.obj_classifier)
        xavier_init(self.rel_feats_embed)

        xavier_init(self.thing_stuff_classifier)
        xavier_init(self.stuff_stuff_classifier)
        xavier_init(self.thing_thing_classifier)
        xavier_init(self.stuff_thing_classifier)

    @staticmethod
    def cal_pair_accuracy(sub_scores, obj_scores, pair_labels):
        sub_labels_pred = torch.argmax(sub_scores, -1)
        obj_labels_pred = torch.argmax(obj_scores, -1)
        sub_accuracy = sub_labels_pred == pair_labels[:, 0]
        obj_accuracy = obj_labels_pred == pair_labels[:, 1]
        all_accuracy = (sub_accuracy + obj_accuracy == 2)
        return sum(all_accuracy) * 1.0 / all_accuracy.shape[0]

    def forward(
        self,
        img,
        img_meta,
        det_result,
        gt_result=None,
        is_testing=False,
        ignore_classes=None,
    ):

        roi_feats, union_feats, det_result = self.prepocess_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        joint_feats, relation_embed, relatedness = self.context_layer(
            img, img_meta, roi_feats, union_feats, det_result
        )

        rel_feats_embed = self.rel_feats_embed(joint_feats)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]

        seperate_feats = joint_feats.view(joint_feats.shape[0], 2, self.hidden_dim)
        sub_feats = seperate_feats[:, 0].contiguous().view(-1, self.hidden_dim)
        obj_feats = seperate_feats[:, 1].contiguous().view(-1, self.hidden_dim)

        refine_sub_scores = self.sub_classifier(sub_feats)
        refine_obj_scores = self.obj_classifier(obj_feats)
        rel_scores = self.rel_classifier(joint_feats)

        pair_pred = torch.stack(
            [torch.argmax(refine_sub_scores, dim=1),
             torch.argmax(refine_obj_scores, dim=1)],
            dim=1)

        split_index = torch.zeros(pair_pred.shape[0]).to(pair_pred)

        thing_thing_index = (pair_pred[:, 0] <= self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
        stuff_stuff_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] > self.num_thing_class)
        stuff_thing_index = (pair_pred[:, 0] > self.num_thing_class) & (pair_pred[:, 1] <= self.num_thing_class)
        split_index[thing_thing_index] = 1
        split_index[stuff_stuff_index] = 2
        split_index[stuff_thing_index] = 3

        thing_stuff_split = joint_feats[split_index == 0]
        thing_thing_split = joint_feats[split_index == 1]
        stuff_stuff_split = joint_feats[split_index == 2]
        stuff_thing_split = joint_feats[split_index == 3]

        thing_stuff_rel_scores = self.thing_stuff_classifier(thing_stuff_split)
        thing_thing_rel_scores = self.thing_thing_classifier(thing_thing_split)
        stuff_stuff_rel_scores = self.stuff_stuff_classifier(stuff_stuff_split)
        stuff_thing_rel_scores = self.stuff_stuff_classifier(stuff_thing_split)

        combine_rel_scores = torch.zeros((pair_pred.shape[0], self.num_predicates)).to(refine_obj_scores)
        combine_rel_scores[split_index == 0] = thing_stuff_rel_scores
        combine_rel_scores[split_index == 1] = thing_thing_rel_scores
        combine_rel_scores[split_index == 2] = stuff_stuff_rel_scores
        combine_rel_scores[split_index == 3] = stuff_thing_rel_scores

        if self.use_bias and not self.use_penalty:
            '''rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())'''
            new_bias = None
            bias = self.freq_bias.index_with_labels(pair_pred.long())
            rel_scores = combine_rel_scores + bias
        elif self.use_penalty:
            new_bias, bias = self.bias_module.index_with_labels(
                pair_pred.long(), epoch=det_result.epoch, max_epochs=det_result.max_epochs)
            rel_scores = combine_rel_scores + new_bias
            if bias is not None:
                det_result.init_rel_scores = combine_rel_scores + bias
            else:
                det_result.init_rel_scores = combine_rel_scores

        det_result.new_bias = new_bias
        det_result.bias = bias

        if self.training:
            target_pair_labels = []
            for i, (rel_pair_idx, target_label) in enumerate(zip(det_result.rel_pair_idxes, det_result.target_labels)):
                target_pair_labels.append(target_label[rel_pair_idx])
            det_result.target_pair_labels = torch.cat(target_pair_labels, dim=-1)
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = None # refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        det_result.rel_feats_embed = rel_feats_embed

        add_for_losses = {}

        if self.training and self.use_triplet_obj_loss:
            add_for_losses['loss_subject_pair'] = self.loss_subject_pair(refine_sub_scores, det_result.target_pair_labels[:, 0]) * 0.5
            add_for_losses['loss_object_pair'] = self.loss_object_pair(refine_obj_scores, det_result.target_pair_labels[:, 1]) * 0.5
            add_for_losses['acc_object_pair'] = self.cal_pair_accuracy(refine_sub_scores, refine_obj_scores, det_result.target_pair_labels)

        det_result.add_losses = add_for_losses

        '''if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(rel_feats, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)'''


        return det_result