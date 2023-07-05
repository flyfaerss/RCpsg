import copy
import itertools

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import build_plugin_layer
from mmdet.core import bbox2roi
from mmdet.models import HEADS, builder
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_neck

from .approaches import (FrequencyBias, PostProcessor, RelationSampler,
                         get_weak_key_rel_labels, FreqBiasModule)
from .approaches.motif_util import obj_edge_vectors


def make_fc(dim_in, hidden_dim):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


@HEADS.register_module()
class RelationHead(BaseModule):
    """The basic class of all the relation head."""
    def __init__(
        self,
        object_classes,
        predicate_classes,
        head_config,
        roi_neck=None,
        pixel_extractor=None,
        bbox_roi_extractor=None,
        relation_roi_extractor=None,
        relation_sampler=None,
        relation_ranker=None,
        dataset_config=None,
        use_bias=False,
        use_statistics=True,
        num_classes=151,
        num_predicates=51,
        loss_object=None,
        loss_relation=None,
        loss_feature=None,
        loss_mask=None,
        loss_dice=None,
        embed_dim=300,
        glove_dir=None,
        transformer=None,
        init_cfg=None,
    ):
        """The public parameters that shared by various relation heads are
        initialized here."""
        super(RelationHead, self).__init__(init_cfg)

        self.use_bias = use_bias
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.embed_dim = embed_dim
        self.glove_dir = glove_dir

        self.transformer = transformer

        # upgrade some submodule attribute to this head
        self.head_config = head_config
        self.use_gt_box = self.head_config.use_gt_box
        self.use_gt_label = self.head_config.use_gt_label
        self.hidden_dim = self.head_config.hidden_dim
        self.init_denoising = self.head_config.init_denoising
        self.use_triplet_obj_loss = self.head_config.use_triplet_obj_loss
        self.feature_extract_method = self.head_config.feature_extract_method
        self.use_penalty = self.head_config.bias_module.use_penalty
        self.hard_index = self.head_config.hard_index
        self.use_peseudo_labels = self.head_config.use_peseudo_labels
        self.use_query_mode = self.head_config.use_query_mode
        self.with_visual_bbox = (bbox_roi_extractor is not None
                                 and bbox_roi_extractor.with_visual_bbox) or (
                                     relation_roi_extractor is not None and
                                     relation_roi_extractor.with_visual_bbox)
        self.with_visual_mask = (bbox_roi_extractor is not None
                                 and bbox_roi_extractor.with_visual_mask) or (
                                     relation_roi_extractor is not None and
                                     relation_roi_extractor.with_visual_mask)
        self.with_visual_point = (bbox_roi_extractor is not None and
                                  bbox_roi_extractor.with_visual_point) or (
                                      relation_roi_extractor is not None and
                                      relation_roi_extractor.with_visual_point)

        self.dataset_config = dataset_config

        self.match_statistic = {}
        self.all_statistic = {}
        self.count_end = 0
        self.image_count = 0
        for i in range(57):
            self.match_statistic[i] = 0
            self.all_statistic[i] = 0


        self.roi_neck = None
        if roi_neck is not None:
            self.roi_neck = build_neck(roi_neck)

        self.pixel_extractor = None
        if self.init_denoising and pixel_extractor is not None:
            self.pixel_extractor = build_plugin_layer(pixel_extractor)[1]

        if self.use_gt_box:
            if self.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        if self.feature_extract_method == 'roi' and bbox_roi_extractor is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
        if self.feature_extract_method == 'roi' and relation_roi_extractor is not None:
            self.relation_roi_extractor = builder.build_roi_extractor(
                relation_roi_extractor)
        if self.feature_extract_method == 'query':
            self.proj_relation = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.hidden_dim * 2), nn.ReLU(),
                make_fc(self.hidden_dim * 2, self.hidden_dim * 2))

            '''self.compress_relation = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.hidden_dim * 2), nn.ReLU(inplace=True),
                make_fc(self.hidden_dim * 2, self.hidden_dim))'''

            self.proj_instance = nn.Sequential(
                make_fc(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
                make_fc(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
                make_fc(self.hidden_dim, self.hidden_dim))

        if relation_sampler is not None:
            relation_sampler.update(dict(use_gt_box=self.use_gt_box))
            self.relation_sampler = RelationSampler(**relation_sampler)

        self.post_processor = PostProcessor()

        # relation ranker: a standard component
        if relation_ranker is not None:
            ranker = relation_ranker.pop('type')
            # self.supervised_form = relation_ranker.pop('supervised_form')
            self.comb_factor = relation_ranker.pop('comb_factor', 0.5)
            self.area_form = relation_ranker.pop('area_form', 'rect')
            loss_ranking_relation = relation_ranker.pop('loss')
            self.loss_ranking_relation = builder.build_loss(
                loss_ranking_relation)
            if loss_ranking_relation.type != 'CrossEntropyLoss':
                num_out = 1
            else:
                num_out = 2
            relation_ranker.update(dict(num_out=num_out))
            self.relation_ranker = eval(ranker)(**relation_ranker)

        self.obj_classes, self.rel_classes = (
            object_classes,
            predicate_classes,
        )
        self.obj_classes.insert(0, '__background__')
        self.rel_classes.insert(0, '__background__')

        assert self.num_classes == len(self.obj_classes)
        assert self.num_predicates == len(self.rel_classes)

        if loss_object is not None:
            self.loss_object = builder.build_loss(loss_object)

        if loss_relation is not None:
            self.loss_relation = builder.build_loss(loss_relation)

        if loss_feature is not None:
            self.loss_feature = builder.build_loss(loss_feature)
            self.relation_embed = obj_edge_vectors(
                self.rel_classes, wv_dir=self.glove_dir, wv_dim=self.embed_dim
            )

        if loss_dice is not None:
            self.loss_dice = builder.build_loss(loss_dice)

        if loss_mask is not None:
            self.loss_mask = builder.build_loss(loss_mask)

        # if self.use_query_mode:
        #     self.relation_query = nn.Embedding(self.num_predicates, self.embed_dims * 2)

        if use_statistics:
            cache_dir = dataset_config['cache']
            predicate_dir = dataset_config['predicate_frequency']
            object_dir = dataset_config['object_frequency']
            self.statistics = torch.load(cache_dir,
                                         map_location=torch.device('cpu'))
            self.freq_matrix = self.statistics['freq_matrix'].float()

            head_num, body_num, tail_num = 8, 20, 28
            self.predicate_distribution = torch.tensor(self.get_frequency_distribution(predicate_dir))
            self.object_distribution = torch.tensor(self.get_frequency_distribution(object_dir))
            self.median_object = torch.median(self.object_distribution)
            a, idx = torch.sort(self.predicate_distribution, descending=False)
            self.body_first = a[head_num + 1]
            self.tail_first = a[head_num + body_num + 1]
            self.weight_ceiling = torch.ones(head_num + body_num + tail_num + 1)
            self.weight_ceiling[self.predicate_distribution > self.tail_first] = torch.sqrt(torch.exp(self.predicate_distribution[
                                        self.predicate_distribution > self.tail_first]) / torch.exp(self.tail_first))
            self.weight_ceiling[self.predicate_distribution < self.tail_first] = torch.sqrt(torch.exp(self.predicate_distribution[
                                        self.predicate_distribution < self.tail_first]) / torch.exp(self.tail_first))
            self.weight_ceiling[self.predicate_distribution < self.body_first] *= torch.sqrt(torch.exp(self.predicate_distribution[
                                        self.predicate_distribution < self.body_first]) / torch.exp(self.body_first))
            self.predicate_weight = torch.ones(head_num + body_num + tail_num + 1).float()

            self.object_weight = torch.ones(len(self.obj_classes))
            '''self.object_weight[self.object_distribution > self.median_object] = torch.sqrt(
                torch.exp(self.object_distribution[self.object_distribution > self.median_object]) / torch.exp(
                    self.median_object))
            self.object_weight[self.object_distribution < self.median_object] = torch.sqrt(
                torch.exp(self.object_distribution[self.object_distribution < self.median_object]) / torch.exp(
                    self.median_object))
            self.object_weight[0] = 1.0'''

            print('\n Statistics loaded!')

        if self.use_bias:
            assert self.with_statistics
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(self.head_config, self.statistics)
        if self.use_penalty:
            self.bias_module = FreqBiasModule(self.head_config, self.statistics)

    @property
    def with_bbox_roi_extractor(self):
        return (hasattr(self, 'bbox_roi_extractor')
                and self.bbox_roi_extractor is not None)

    @property
    def with_relation_roi_extractor(self):
        return (hasattr(self, 'relation_roi_extractor')
                and self.relation_roi_extractor is not None)

    @property
    def with_statistics(self):
        return hasattr(self, 'statistics') and self.statistics is not None

    @property
    def with_bias(self):
        return hasattr(self, 'freq_bias') and self.freq_bias is not None

    @property
    def with_loss_object(self):
        return hasattr(self, 'loss_object') and self.loss_object is not None

    @property
    def with_loss_relation(self):
        return hasattr(self,
                       'loss_relation') and self.loss_relation is not None

    @property
    def with_loss_feature(self):
        return hasattr(self,
                       'loss_feature') and self.loss_feature is not None

    @property
    def with_relation_ranker(self):
        return hasattr(self,
                       'relation_ranker') and self.relation_ranker is not None

    def init_weights(self):
        if self.with_bbox_roi_extractor:
            self.bbox_roi_extractor.init_weights()
        if self.with_relation_roi_extractor:
            self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

    def get_relation_feature(self, obj_querys, rel_pair_idxes):
        batch = len(obj_querys)
        all_pairs = []
        for idx, (obj_query, rel_pair_idx) in enumerate(zip(obj_querys, rel_pair_idxes)):
            per_image_relation_feature = torch.cat((obj_query[rel_pair_idx[:, 0]], obj_query[rel_pair_idx[:, 1]]), dim=-1)
            all_pairs.append(per_image_relation_feature)
        all_pairs = torch.cat(all_pairs)
        all_pairs = self.proj_relation(all_pairs)

        return all_pairs

    def get_relation_feature_plus(self, obj_feats, rel_pair_idxes, num_obj):
        all_pairs = []
        start_idx = 0
        for idx, (rel_pair_idx) in enumerate(rel_pair_idxes):
            per_image_relation_feature = torch.cat((obj_feats[rel_pair_idx[:, 0] + start_idx], obj_feats[rel_pair_idx[:, 1] + start_idx]), dim=-1)
            start_idx += num_obj[idx]
            all_pairs.append(per_image_relation_feature)
        all_pairs = torch.cat(all_pairs)
        all_pairs = self.proj_relation(all_pairs)

        return all_pairs

    def cal_union_masks(self, rel_pair_idxes, target_masks, pad_shape):
        target_rel_masks = []
        height, width = pad_shape

        for (target_mask, rel_pair_idx) in zip(target_masks, rel_pair_idxes):
            union_masks = target_mask[rel_pair_idx[:, 0]] | target_mask[rel_pair_idx[:, 1]]
            img_height, img_width = union_masks.shape[1:]
            pad = (0, int(width - img_width), 0, int(height - img_height))
            union_masks = F.pad(union_masks, pad, 'constant', 0)
            target_rel_masks.append(union_masks)

        return target_rel_masks

    def prepocess_features(self, img, img_meta, det_result, gt_result):

        # obj_query = None
        bboxes, masks, obj_query = (
            det_result.bboxes,
            det_result.masks,
            det_result.obj_feats_query,
        )

        if gt_result is not None and gt_result.rels is not None:
            sample_function = self.relation_sampler.premask_relsample
            sample_res = sample_function(det_result, gt_result)
            rel_labels, rel_pair_idxes, rel_matrix =  sample_res
        else:
            rel_labels, rel_matrix = None, None
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(
                det_result)

        '''if gt_result is not None and det_result.epoch == 0:
            for i, (gt_rel, match_gt_rel) in enumerate(zip(gt_result.rel_labels, rel_labels)):
                self.image_count += 1
                for item in gt_rel:
                    self.all_statistic[int(item)] += 1
                for item in match_gt_rel:
                    self.match_statistic[int(item)] += 1
                    if item == 0:
                        break
            # print(self.image_count, self.all_statistic)

        if det_result.epoch == 1 and self.count_end == 0:
            self.count_end = 1
            all_dir = '/home/sylvia/yjr/sgg/OpenPSG/data/psg/all_statistic.txt'
            match_dir = '/home/sylvia/yjr/sgg/OpenPSG/data/psg/match_statistic.txt'
            with open(all_dir, 'w') as f:
                f.write(str(self.all_statistic))
            f.close()
            with open(match_dir, 'w') as f:
                f.write(str(self.match_statistic))
            f.close()'''


        det_result.rel_pair_idxes = rel_pair_idxes
        det_result.relmaps = rel_matrix
        det_result.target_rel_labels = rel_labels
        det_result.target_key_rel_labels = None

        # extract the unary roi features and union roi features.
        if self.feature_extract_method == 'roi':
            rois = bbox2roi(bboxes)
            # merge image-wise masks or points
            if masks is not None:
                masks = list(itertools.chain(*masks))
            # masks = None
            if self.roi_neck is not None:
                img = self.roi_neck(img)
            roi_feats = self.bbox_roi_extractor(img,
                                                img_meta,
                                                rois,
                                                masks=masks,
                                                points=None,)
            union_feats = self.relation_roi_extractor(img,
                                                      img_meta,
                                                      rois,
                                                      rel_pair_idx=rel_pair_idxes,
                                                      masks=masks,
                                                      points=None)

            return roi_feats + union_feats + (det_result, )
        elif self.feature_extract_method == 'query':
            if self.init_denoising:
                roi_feats = torch.cat(obj_query, dim=0)
                roi_feats = self.proj_instance(roi_feats)
                num_obj = [x.shape[0] for x in obj_query]
                union_feats = self.get_relation_feature_plus(roi_feats, rel_pair_idxes, num_obj)
                # labeled_length = [sum(x>0) for x in rel_labels]
                # used_rel_pair_idxes = [x[:] for x in rel_pair_idxes]
                if self.training:
                    per_len = [x.shape[0] for x in rel_pair_idxes]
                    split_union_feats = torch.split(union_feats, per_len, dim=0)
                    neg_num = 10
                    pos_len = []
                    part_union_feats = []
                    valid_rel_pair_idxes = []
                    for i, rel_label in enumerate(rel_labels):
                        num_total_rel = rel_label.shape[0]
                        num_pos_rel = sum(rel_label > 0)
                        num_neg_rel = num_total_rel - num_pos_rel
                        neg_rand_idx = torch.randperm(num_neg_rel).to(num_pos_rel) + num_pos_rel
                        neg_valid_num = min(neg_num, num_neg_rel)
                        pos_len.append(neg_valid_num + num_pos_rel)
                        rand_idx = torch.cat((torch.arange(num_pos_rel).to(neg_rand_idx), neg_rand_idx[:neg_valid_num]), dim=0)
                        part_union_feats.append(split_union_feats[i][rand_idx])
                        valid_rel_pair_idxes.append(rel_pair_idxes[i][rand_idx])

                    part_union_feats = torch.cat(part_union_feats)

                    split_union_feats = self.compress_relation(part_union_feats)
                    # per_len = [x.shape[0] for x in rel_pair_idxes]
                    split_union_feats = torch.split(split_union_feats, pos_len, dim=0)
                    # if len(img) == 4:
                    #     img = img[1:]
                    mask_features, multi_scale_memorys = self.pixel_extractor(img)
                    rel_masks = []
                    for i in range(mask_features.shape[0]):
                        rel_masks.append(torch.einsum('qc, chw->qhw', split_union_feats[i], mask_features[i]))

                    det_result.rel_masks = torch.cat(rel_masks, dim=0)

                    '''rel_masks = torch.cat(rel_masks, dim=0)
                    small_size = (rel_masks.shape[1] // 2, rel_masks.shape[2] // 2)
                    det_result.rel_masks = F.interpolate(
                            rel_masks.unsqueeze(0),
                            size=small_size,
                            mode='nearest').squeeze(0)'''

                    target_rel_masks = self.cal_union_masks(valid_rel_pair_idxes, det_result.target_masks, img_meta[0]['batch_input_shape'])
                    target_rel_masks = torch.cat(target_rel_masks, dim=0)
                    target_rel_masks = F.interpolate(
                        target_rel_masks.unsqueeze(0).float(),
                        size=det_result.rel_masks.shape[1:],
                        mode='nearest').squeeze(0).long()
                    det_result.target_rel_masks = target_rel_masks.to(det_result.rel_masks)

                return roi_feats, union_feats, det_result
            else:
                roi_feats = torch.cat(obj_query, dim=0)
                # roi_feats = self.proj_instance(roi_feats)
                union_feats = self.get_relation_feature(obj_query, rel_pair_idxes)
                # roi_feats = self.proj_instance(roi_feats)

                return roi_feats, union_feats, det_result

    def frontend_features(self, img, img_meta, det_result, gt_result):
        bboxes, masks, points = (
            det_result.bboxes,
            det_result.masks,
            copy.deepcopy(det_result.points),
        )

        # train/val or: for finetuning on the dataset without
        # relationship annotations
        if gt_result is not None and gt_result.rels is not None:
            if self.mode in ['predcls', 'sgcls']:
                sample_function = self.relation_sampler.gtbox_relsample
            else:
                sample_function = self.relation_sampler.detect_relsample

            sample_res = sample_function(det_result, gt_result)

            if len(sample_res) == 4:
                rel_labels, rel_pair_idxes, rel_matrix, \
                    key_rel_labels = sample_res
            else:
                rel_labels, rel_pair_idxes, rel_matrix = sample_res
                key_rel_labels = None
        else:
            rel_labels, rel_matrix, key_rel_labels = None, None, None
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(
                det_result)

        '''gt_on, gt_over, train_on, train_over = 0, 0, 0, 0
        for item in gt_result.rel_labels:
            gt_over += sum(item == 1)
            gt_on += sum(item == 4)
        for item in rel_labels:
            train_over += sum(item == 1)
            train_on += sum(item == 4)
        print('gt_on: {}; gt_over: {}; train_on: {}; train_over: {}'.format(gt_on, gt_over, train_on, train_over))'''


        det_result.rel_pair_idxes = rel_pair_idxes
        det_result.relmaps = rel_matrix
        det_result.target_rel_labels = rel_labels
        det_result.target_key_rel_labels = key_rel_labels

        rois = bbox2roi(bboxes)
        # merge image-wise masks or points
        if masks is not None:
            masks = list(itertools.chain(*masks))
        if points is not None:
            aug_points = []
            for pts_list in points:
                for pts in pts_list:
                    pts = pts.view(-1, 2)  # (:, [x, y])
                    pts += torch.from_numpy(
                        np.random.normal(0, 0.02, size=pts.shape)).to(pts)
                    # pts -= torch.mean(pts, dim=0, keepdim=True)
                    pts /= torch.max(torch.sqrt(torch.sum(pts**2, dim=1)))
                    aug_points.append(pts)
            points = aug_points

        feature_extract_method = True

        # extract the unary roi features and union roi features.
        if feature_extract_method:
            roi_feats = self.bbox_roi_extractor(img,
                                                img_meta,
                                                rois,
                                                masks=masks,
                                                points=points)
            union_feats = self.relation_roi_extractor(img,
                                                      img_meta,
                                                      rois,
                                                      rel_pair_idx=rel_pair_idxes,
                                                      masks=masks,
                                                      points=points)
        else:
            roi_feats = self.bbox_roi_extractor(img,
                                                img_meta,
                                                rois,
                                                masks=masks,
                                                points=points)
            union_feats = self.relation_roi_extractor(img,
                                                      img_meta,
                                                      rois,
                                                      rel_pair_idx=rel_pair_idxes,
                                                      masks=masks,
                                                      points=points)

        return roi_feats + union_feats + (det_result, )
        # return roi_feats, union_feats, (det_result,)

    def forward(self, **kwargs):
        raise NotImplementedError

    def relation_ranking_forward(self, input, det_result, gt_result, num_rels,
                                 is_testing):
        # predict the ranking

        # tensor
        ranking_scores = self.relation_ranker(
            input.detach(), det_result, self.relation_roi_extractor.union_rois)

        # (1) weak supervision, KLDiv:
        if self.loss_ranking_relation.__class__.__name__ == 'KLDivLoss':
            if not is_testing:  # include training and validation
                # list form
                det_result.target_key_rel_labels = get_weak_key_rel_labels(
                    det_result, gt_result, self.comb_factor, self.area_form)
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = ranking_scores.split(num_rels, 0)
            else:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = torch.sigmoid(ranking_scores).split(num_rels,
                                                                     dim=0)

        # (2) CEloss: the predicted one is the binary classification, 2 columns
        if self.loss_ranking_relation.__class__.__name__ == 'CrossEntropyLoss':
            if not is_testing:
                det_result.target_key_rel_labels = torch.cat(
                    det_result.target_key_rel_labels, dim=-1)
            else:
                ranking_scores = (F.softmax(ranking_scores,
                                            dim=-1)[:, 1].view(-1).split(
                                                num_rels, 0))
        # Margin loss, DR loss
        elif self.loss_ranking_relation.__class__.__name__ == 'SigmoidDRLoss':
            if not is_testing:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = ranking_scores.split(num_rels, 0)
            else:
                ranking_scores = ranking_scores.view(-1)
                ranking_scores = torch.sigmoid(ranking_scores).split(num_rels,
                                                                     dim=0)

        det_result.ranking_scores = ranking_scores
        return det_result

    def auxiliary_mask_loss(self, pred_rel_masks, target_rel_masks):
        num_total_masks, h, w = pred_rel_masks.shape
        loss_dice = self.loss_dice(pred_rel_masks.reshape(num_total_masks, -1), target_rel_masks.reshape(num_total_masks, -1), avg_factor=num_total_masks)

        # pred_rel_masks = pred_rel_masks.reshape(-1)
        # target_rel_masks = target_rel_masks.reshape(-1)
        # loss_mask = self.loss_mask(pred_rel_masks, target_rel_masks, avg_factor=num_total_masks * h * w)

        return loss_dice # , loss_mask

    def loss(self, det_result):
        (
            ori_obj_preds,
            obj_scores,
            rel_scores,
            init_rel_scores,
            target_labels,
            target_rel_labels,
            add_for_losses,
            resistance_bias,
            rel_feats_embed,
            epoch,
            max_epochs,
            head_spec_losses,
        ) = (
            det_result.dists,
            det_result.refine_scores,
            det_result.rel_scores,
            det_result.init_rel_scores,
            det_result.target_labels,
            det_result.target_rel_labels,
            det_result.add_losses,
            det_result.resistance_bias,
            det_result.rel_feats_embed,
            det_result.epoch,
            det_result.max_epochs,
            det_result.head_spec_losses,
        )

        losses = dict()
        if self.with_loss_object and obj_scores is not None:
            # fix: the val process
            if isinstance(target_labels, (tuple, list)):
                target_labels = torch.cat(target_labels, dim=-1)
            if isinstance(obj_scores, (tuple, list)):
                obj_scores = torch.cat(obj_scores, dim=0)

            # losses['loss_object'] = self.loss_object(obj_scores, target_labels, self.object_weight.to(obj_scores.device)) * 0.5
            losses['loss_object'] = self.loss_object(obj_scores, target_labels) * 0.5
            losses['acc_object'] = accuracy(obj_scores, target_labels)

        '''if isinstance(target_labels, (tuple, list)):
            target_labels = torch.cat(target_labels, dim=-1)
        if isinstance(ori_obj_preds, (tuple, list)):
            ori_obj_preds = torch.cat(ori_obj_preds, dim=0)
        losses['acc_object'] = accuracy(ori_obj_preds, target_labels)'''

        loss_weight = torch.tensor((epoch + 1) / float(max_epochs + 1)).to(rel_feats_embed.device)
        '''self.predicate_weight[self.predicate_distribution < self.body_first] = self.weight_ceiling[
                            self.predicate_distribution < self.body_first] * (epoch / float(1.2 * max_epochs))
        self.predicate_weight[self.predicate_distribution >= self.body_first] = self.weight_ceiling[
                            self.predicate_distribution >= self.body_first] * (0.2 + epoch / float(1.2 * max_epochs))
        self.predicate_weight[self.predicate_distribution >= self.tail_first] = self.weight_ceiling[
                            self.predicate_distribution >= self.tail_first] * (1.0 + epoch / float(1.2 * max_epochs))'''
        # self.predicate_weight[0] = 0.1 # 0.01

        if self.with_loss_relation and rel_scores is not None:
            if isinstance(target_rel_labels, (tuple, list)):
                target_rel_labels = torch.cat(target_rel_labels, dim=-1)
            if isinstance(rel_scores, (tuple, list)):
                rel_scores = torch.cat(rel_scores, dim=0)
            '''masked_bg_rel_num = max(len(target_rel_labels) - 4 * len(torch.nonzero(target_rel_labels)), 0)
            bg_index = torch.nonzero(target_rel_labels == 0).squeeze()
            shuffle_bg_index = bg_index[torch.randperm(len(bg_index))]
            masked_bg_index = shuffle_bg_index[:masked_bg_rel_num]
            target_rel_labels[masked_bg_index] = -1'''
            # losses['loss_relation'] = self.loss_relation(rel_scores, target_rel_labels, self.predicate_weight.to(rel_feats_embed.device)) * loss_weight
            if self.use_peseudo_labels:
                pred_label = torch.argmax(rel_scores, dim=1)
                count = 0
                for i in range(pred_label.shape[0]):
                    if pred_label[i] in self.hard_index and target_rel_labels[i] == 0:
                        # rest_score = true_dist[i, 0] + true_dist[i, pred_label[i]]
                        target_rel_labels[i] = pred_label[i]  # rest_score / 2.0 # += 0.2
                        count += 1
                if count != 0:
                    print(count)
                losses['loss_relation'] = self.loss_relation(rel_scores, target_rel_labels)
            else:
                losses['loss_relation'] = self.loss_relation(rel_scores, target_rel_labels)  # * loss_weight
            # losses['loss_relation'] = self.loss_relation(rel_scores, target_rel_labels, self.hard_index, resistance_bias, 2.0 - epoch / max_epochs) # * loss_weight
            # temp_pred =
            losses['acc_relation'] = accuracy(rel_scores[target_rel_labels > 0], target_rel_labels[target_rel_labels > 0])
            if self.use_penalty:
                losses['acc_bias_relation'] = accuracy(init_rel_scores[target_rel_labels > 0],
                                                  target_rel_labels[target_rel_labels > 0])

        if self.with_loss_feature and (self.with_loss_relation and rel_scores is not None):
            losses['loss_feature'] = self.loss_feature(rel_feats_embed, target_rel_labels,
                                                       self.relation_embed.to(rel_feats_embed.device),
                                                       self.predicate_weight.to(rel_feats_embed.device)) * (1.0 - loss_weight)
            losses['feature_weight'] = 1.0 - loss_weight

        if self.init_denoising:
            losses['loss_dice_auxiliary'] = self.auxiliary_mask_loss(det_result.rel_masks, det_result.target_rel_masks)

        if self.with_relation_ranker:
            target_key_rel_labels = det_result.target_key_rel_labels
            ranking_scores = det_result.ranking_scores

            avg_factor = (torch.nonzero(
                target_key_rel_labels != -1).view(-1).size(0) if isinstance(
                    target_key_rel_labels, torch.Tensor) else None)
            losses['loss_ranking_relation'] = self.loss_ranking_relation(
                ranking_scores, target_key_rel_labels, avg_factor=avg_factor)
            # if self.supervised_form == 'weak':
            #     # use the KLdiv loss: the label is the soft distribution
            #     bs = 0
            #     losses['loss_ranking_relation'] = 0
            #     for ranking_score, target_key_rel_label in zip(ranking_scores, target_key_rel_labels):
            #         bs += ranking_score.size(0)
            #         losses['loss_ranking_relation'] += torch.nn.KLDivLoss(reduction='none')(F.log_softmax(ranking_score, dim=-1),
            #                                                                     target_key_rel_label).sum(-1)
            #     losses['loss_ranking_relation'] = losses['loss_ranking_relation'] / bs
            # else:
            #     #TODO: firstly try the CE loss function, or you may try the margin loss
            #     #TODO: Check the margin loss
            #     #loss_func = builder.build_loss(self.loss_ranking_relation)
            #     losses['loss_ranking_relation'] = self.loss_ranking_relation(ranking_scores, target_key_rel_labels)

        if add_for_losses is not None:
            for loss_key, loss_item in add_for_losses.items():
                if isinstance(loss_item, list):  # loss_vctree_binary
                    loss_ = [
                        F.binary_cross_entropy_with_logits(l[0], l[1])
                        for l in loss_item
                    ]
                    loss_ = sum(loss_) / len(loss_)
                    losses[loss_key] = loss_
                elif isinstance(loss_item, tuple):
                    if isinstance(loss_item[1], (list, tuple)):
                        target = torch.cat(loss_item[1], -1)
                    else:
                        target = loss_item[1]
                    losses[loss_key] = F.cross_entropy(loss_item[0], target)
                else:
                    raise NotImplementedError

        if head_spec_losses is not None:
            # this losses have been calculated in the specific relation head
            losses.update(head_spec_losses)

        return losses

    def get_result(self, det_result, scale_factor, rescale, key_first=False):
        """for test forward.

        :param det_result:
        :return:
        """
        result = self.post_processor(det_result, key_first=key_first, freq_matrix=self.freq_matrix)

        for k, v in result.__dict__.items():
            if (k != 'add_losses' and k != 'head_spec_losses' and v is not None
                    and len(v) > 0):
                _v = v[0]  # remove the outer list
                if isinstance(_v, torch.Tensor):
                    result.__setattr__(k, _v.cpu().numpy())
                elif isinstance(_v, list):  # for mask
                    result.__setattr__(k, [__v.cpu().numpy() for __v in _v])
                else:
                    result.__setattr__(k, _v)  # e.g., img_shape, is a tuple

        if rescale:
            if result.bboxes is not None:
                result.bboxes[:, :4] = result.bboxes[:, :4] / scale_factor
            if result.refine_bboxes is not None:
                result.refine_bboxes[:, :
                                     4] = result.refine_bboxes[:, :
                                                               4] / scale_factor

            if result.masks is not None:
                resize_masks = []
                for bbox, mask in zip(result.refine_bboxes, result.masks):
                    _bbox = bbox.astype(np.int32)
                    w = max(_bbox[2] - _bbox[0] + 1, 1)
                    h = max(_bbox[3] - _bbox[1] + 1, 1)
                    resize_masks.append(
                        mmcv.imresize(mask.astype(np.uint8), (w, h)))
                result.masks = resize_masks

            if result.points is not None:
                resize_points = []
                for points in result.points:
                    resize_points.append(points / scale_factor)
                result.points = resize_points

        # if needed, adjust the form for object detection evaluation
        result.formatted_bboxes, result.formatted_masks = [], []

        if result.refine_bboxes is None:
            result.formatted_bboxes = [
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.num_classes - 1)
            ]
        else:
            result.formatted_bboxes = [
                result.refine_bboxes[result.refine_labels == i + 1, :]
                for i in range(self.num_classes - 1)
            ]

        if result.masks is None:
            result.formatted_masks = [[] for i in range(self.num_classes - 1)]
        else:
            result.formatted_masks = [[] for i in range(self.num_classes - 1)]
            for i in range(len(result.masks)):
                result.formatted_masks[result.refine_labels[i] - 1].append(
                    result.masks[i])

        # to save the space, drop the saliency maps, if it exists
        if result.saliency_maps is not None:
            result.saliency_maps = None

        return result

    def process_ignore_objects(self, input, ignore_classes):
        """An API used in inference stage for processing the data when some
        object classes should be ignored."""
        ignored_input = input.clone()
        ignored_input[:, ignore_classes] = 0.0
        return ignored_input

    def get_frequency_distribution(self, dir):
        with open(dir, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
        data = data[1:-1].split()
        ls = map(float, data)
        frequency_informative = list(ls)
        padding = [1.0]
        padding.extend(frequency_informative)

        return padding
