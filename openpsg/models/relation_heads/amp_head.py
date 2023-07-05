import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from mmdet.models import HEADS

from .approaches import AFEContext
from .relation_head import RelationHead


@HEADS.register_module()
class AMPHead(RelationHead):
    def __init__(self, **kwargs):
        super(AMPHead, self).__init__(**kwargs)

        self.context_layer = AFEContext(
            self.head_config,
        )

        self.use_obj_refine = self.head_config.use_obj_refine
        self.hidden_dim = self.head_config.hidden_dim

        # post classification
        self.rel_classifier = nn.Linear(self.hidden_dim, self.num_predicates)
        self.obj_classifier = nn.Linear(self.hidden_dim, self.num_classes)

        self.embed_dim = 300
        self.rel_feats_embed = nn.Linear(self.hidden_dim, self.embed_dim)

        if self.use_obj_refine:
            self.inst_feats_embed = nn.Linear(self.hidden_dim, self.embed_dim)

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        xavier_init(self.rel_classifier)
        xavier_init(self.obj_classifier)
        xavier_init(self.rel_feats_embed)

        if self.use_obj_refine:
            xavier_init(self.inst_feats_embed)

    def forward(
        self,
        img,
        img_meta,
        det_result,
        gt_result=None,
        is_testing=False,
        ignore_classes=None,

        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        loss_weight=None,
        predicate_weight=None,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # obj_feats : refined object feature (number of object in batch, 512)
        # rel_feats : refined relation feature (number of relation in batch(fully connected), 512)
        # pre_cls_logits : confidence for each predicate class in each relation in 3 iteration list(3)
        # each : (number of relation in batch(fully connected), 51) [the last element is for global confidence]
        # relatedness : (batch_size, the number of object in an image, the number of object in an image, iteration:3)
        # this data is just for test

        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        obj_feats, rel_feats, relation_embed, relatedness = self.context_layer(
            roi_feats, union_feats, inst_proposals, rel_pair_idxs, rel_labels, logger
        )

        rel_feats_embed = self.rel_feats_embed(rel_feats)
        # inst_feats_embed = self.inst_feats_embed(obj_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]

        refine_obj_scores = self.obj_classifier(obj_feats)
        rel_scores = self.rel_classifier(rel_feats)
        obj_preds =

        pair_preds = []
        for pair_idx, obj_pred in zip(
                det_result.rel_pair_idxes, obj_preds):
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))
        pair_pred = torch.cat(pair_preds, dim=0)

        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())

        if self.use_obj_refine:
            if self.training:
                det_result.target_labels = torch.cat(det_result.target_labels,
                                                     dim=-1)
                det_result.target_rel_labels = torch.cat(
                    det_result.target_rel_labels, dim=-1)
            else:
                refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
                rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        det_result.rel_feats_embed = rel_feats_embed

        add_for_losses = {}
        det_result.add_losses = add_for_losses

        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(rel_feats, det_result,
                                                       gt_result, num_rels,
                                                       is_testing)


        return det_result