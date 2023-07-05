# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, SingleStageDetector
from mmcv.runner.fp16_utils import auto_fp16

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap


def triplet2Result(triplets, use_mask, eval_pan_rels=True):
    if use_mask:
        bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
            r_labels, r_dists, pan_masks, rels, pan_labels, r_scores, all_scores, all_labels, all_masks, \
            rel_pairs_index, all_rels, final_r_labels, final_r_dists \
            = triplets
        if isinstance(bboxes, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            rel_pairs = rel_pairs.detach().cpu().numpy()
            complete_r_labels = complete_r_labels.detach().cpu().numpy()
            complete_r_dists = complete_r_dists.detach().cpu().numpy()
            r_labels = r_labels.detach().cpu().numpy()
            r_dists = r_dists.detach().cpu().numpy()
        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.detach().cpu().numpy()
            pan_rel_pairs = pan_rel_pairs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            pan_masks = pan_masks.detach().cpu().numpy()
            rels = rels.detach().cpu().numpy()
            pan_labels = pan_labels.detach().cpu().numpy()
        if eval_pan_rels:
            return Result(refine_bboxes=bboxes,
                        labels=all_labels.cpu().numpy()+1,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=rel_pairs_index.cpu().numpy(),# elif not pan: rel_pairs,
                        rel_dists=final_r_dists.cpu().numpy(),
                        rel_labels=final_r_labels.cpu().numpy(),
                        pan_results=pan_seg,
                        seg_scores=all_scores,
                        triplet_scores=r_scores,
                        masks=all_masks,
                        rels=all_rels.cpu().numpy())
            '''return Result(refine_bboxes=bboxes,
                        labels=pan_labels+1,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=pan_rel_pairs,# elif not pan: rel_pairs,
                        rel_dists=r_dists,
                        rel_labels=r_labels,
                        pan_results=pan_seg,
                        masks=pan_masks,
                        rels=rels)'''
        else:
            return Result(refine_bboxes=bboxes,
                        labels=labels,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=rel_pairs,
                        rel_dists=complete_r_dists,
                        rel_labels=complete_r_labels,
                        pan_results=pan_seg,
                        masks=masks)
    else:
        bboxes, labels, rel_pairs, r_labels, r_dists = triplets
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        rel_pairs = rel_pairs.detach().cpu().numpy()
        r_labels = r_labels.detach().cpu().numpy()
        r_dists = r_dists.detach().cpu().numpy()
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=None),
            rel_pair_idxes=rel_pairs,
            rel_dists=r_dists,
            rel_labels=r_labels,
            pan_results=None,
        )


@DETECTORS.register_module()
class PSGTr(SingleStageDetector):
    def __init__(self,
                 backbone,
                 # neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PSGTr, self).__init__(backbone, None, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def set_epoch(self, epoch):
        self.backbone.epoch = epoch

    def set_max_epochs(self, max_epochs):
        self.backbone.max_epochs = max_epochs

    @auto_fp16(apply_to=('img',))
    def forward_train(self,
                      img,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        if self.bbox_head.use_mask:
            BS, C, H, W = img.shape
            new_gt_masks = []
            for each in gt_masks:
                mask = torch.tensor(each.to_ndarray(), device=x[0].device)
                _, h, w = mask.shape
                padding = (0, W - w, 0, H - h)
                mask = F.interpolate(F.pad(mask, padding).unsqueeze(1),
                                     size=(H // 2, W // 2),
                                     mode='nearest').squeeze(1)
                # mask = F.pad(mask, padding)
                new_gt_masks.append(mask)

            gt_masks = new_gt_masks

        losses = self.bbox_head.forward_train(x, img_metas, gt_rels, gt_bboxes,
                                              gt_labels, gt_masks,
                                              gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat,
                                                  img_metas,
                                                  rescale=rescale)


        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)

        if True:
            pan_results, pan_masks, pair_idxes, inst_idxes, idxes_map = self.backward_panoptic_inference(sg_results[0])
            sg_results[0].pan_results = pan_results
            sg_results[0].masks = pan_masks
            # masks, refine_bboxes, refine_labels, rel_dists, rel_pair_idxes, rels
            sg_results[0].refine_bboxes = sg_results[0].refine_bboxes[inst_idxes, :]
            # det_result.refine_labels = det_result.refine_labels[inst_idxes]
            sg_results[0].refine_labels = sg_results[0].labels[inst_idxes]
            sg_results[0].rel_dists = sg_results[0].rel_dists[pair_idxes, :]
            sg_results[0].rel_pair_idxes = idxes_map[sg_results[0].rel_pair_idxes[pair_idxes, :]]
            sg_results[0].rels = sg_results[0].rels[pair_idxes, :]
            sg_results[0].rels[:, :2] = sg_results[0].rel_pair_idxes
            # print(np.max(det_result.rel_pair_idxes))

        return sg_results

    def backward_panoptic_inference(self, det_result):
        seg_scores = det_result.seg_scores
        init_pan_masks = det_result.masks.cpu().numpy()
        refine_obj_labels = det_result.labels - 1
        triplet_scores = det_result.triplet_scores.cpu().numpy()
        relations = det_result.rels

        stuff_equiv_classes = defaultdict(lambda: [])
        thing_classes = defaultdict(lambda: [])
        thing_dedup = defaultdict(lambda: [])
        for k, label in enumerate(refine_obj_labels):
            if label.item() >= 80:
                stuff_equiv_classes[label.item()].append(k)
            else:
                thing_classes[label.item()].append(k)

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

        syn_scores = triplet_scores

        id_unique = 1
        instance_cover_sequence = []
        sorted_instance_index = []
        sorted_instance_labels = []
        area_list = {}
        # 0: instance index 1:
        # results = np.zeros((2, *init_pan_masks.shape[-2:]))
        pan_index = np.zeros((init_pan_masks.shape[-2:]))
        pan_results = np.zeros((init_pan_masks.shape[-2:]))

        used_flag = np.zeros(init_pan_masks.shape[0])
        length = min(len(syn_scores), 50)

        idx2score = {}
        idx2relscore = {}

        for i, rel in enumerate(relations):
            if used_flag[rel[0]] == 0:
                temp_score = triplet_scores[i] * seg_scores[rel[0]]
                idx2score[rel[0]] = temp_score
                idx2relscore[rel[0]] = triplet_scores[i]
                used_flag[rel[0]] = 1
            if used_flag[rel[1]] == 0:
                temp_score = triplet_scores[i] * seg_scores[rel[1]]
                idx2score[rel[1]] = temp_score
                idx2relscore[rel[1]] = triplet_scores[i]
                used_flag[rel[1]] = 1

        # sorted_score = sorted(idx2score.items(), key=lambda x: x[1], reverse=True)

        # sorted_score_1 = np.array([x[1].cpu() for x in sorted_score])
        # sorted_score_1_num_1 = np.sum(seg_scores.cpu().numpy() > 0.5)
        # sorted_score_1_num_2 = np.sum(seg_scores.cpu().numpy() > 0.8)
        # instance_cover_sequence = [item[0] + 1 for item in sorted_score if item[1] > 0.04]
        # instance_cover_sequence = [item[0] + 1 for item in sorted_score]

        '''for i in range(len(syn_scores)):
            sub_index, obj_index = relations[i, :2] + 1
            if used_flag[sub_index - 1] == 0:
                instance_cover_sequence.append(sub_index)
                used_flag[sub_index - 1] = 1
            if used_flag[obj_index - 1] == 0:
                instance_cover_sequence.append(obj_index)
                used_flag[obj_index - 1] = 1'''

        # instance_cover_num = np.sum(seg_scores.cpu().numpy() > 0.2)
        # instance_cover_sequence = np.arange(instance_cover_num) + 1

        # instance_cover_sequence = np.arange(init_pan_masks.shape[0]) + 1
        instance_cover_num = np.sum(seg_scores.cpu().numpy() > 0.3)
        sorted_seg_scores, sorted_idxes = torch.sort(seg_scores, descending=True)
        instance_cover_sequence = sorted_idxes[:instance_cover_num].cpu().numpy() + 1
        # instance_cover_sequence = sorted_idxes.cpu().numpy() + 1
        label_set = []

        for inst_index in instance_cover_sequence:
            # special for universal panoptic segmentation framework
            '''if refine_obj_labels[inst_index - 1] >= 80 and (refine_obj_labels[inst_index - 1] in sorted_instance_labels):
                old_index = sorted_instance_labels.index(refine_obj_labels[inst_index - 1])
                _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                pan_index[_mask] = sorted_instance_index[old_index]
                pan_results[_mask] = refine_obj_labels[inst_index - 1]
                # old2map[inst_index - 1] = sorted_instance_index[old_index] - 1
                continue'''
            if inst_index - 1 not in idx2relscore.keys():
                idx2relscore[inst_index - 1] = 0.0
            pan_index, pan_results, area_list, cover_flag, id_unique = self.get_cover_mask(pan_index, pan_results,
                                                                                           init_pan_masks, area_list,
                                                                                           inst_index,
                                                                                           refine_obj_labels,
                                                                                           idx2relscore[inst_index - 1],
                                                                                           id_unique)
            if cover_flag:
                sorted_instance_index.append(inst_index)
                if refine_obj_labels[inst_index - 1] >= 80:
                    sorted_instance_labels.append(refine_obj_labels[inst_index - 1])
                else:
                    sorted_instance_labels.append(refine_obj_labels[inst_index - 1] + (id_unique - 1) * INSTANCE_OFFSET)

        pan_results[pan_results == 0] = 133
        sorted_instance_labels = np.array(sorted_instance_labels)
        sorted_instance_index = list(np.array(sorted_instance_index) - 1)

        idxes_map = np.zeros(init_pan_masks.shape[0], dtype=np.int) - 1
        if len(sorted_instance_index) != 0:
            idxes_map[sorted_instance_index] = np.arange(len(sorted_instance_index))

        used_index = np.zeros(init_pan_masks.shape[0], dtype=np.bool)
        if len(sorted_instance_index) != 0:
            used_index[sorted_instance_index] = True

        pan_masks = pan_results[None] == sorted_instance_labels[:, None, None]

        sub_used_list, obj_used_list = used_index[relations[:, 0]], used_index[relations[:, 1]]
        pair_used_idxes = sub_used_list & obj_used_list

        return pan_results, pan_masks, pair_used_idxes, sorted_instance_index, idxes_map

    def get_cover_mask(self, pan_index, pan_results, init_pan_masks, area_list, inst_index, inst_labels, rel_score,
                       id_unique):

        cover_flag = False
        cover_statistic, cover_index_list, cover_area = self.get_covered_area(pan_index, init_pan_masks[inst_index - 1])
        if len(cover_index_list) == 0:
            pan_index[init_pan_masks[inst_index - 1]] = inst_index
            if inst_labels[inst_index - 1] < 80:
                pan_results[init_pan_masks[inst_index - 1]] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                id_unique += 1
            else:
                pan_results[init_pan_masks[inst_index - 1]] = inst_labels[inst_index - 1]
            area_list[inst_index] = np.sum(init_pan_masks[inst_index - 1])
            cover_flag = True
        else:
            cover_ratio = cover_area * 1.0 / np.sum(init_pan_masks[inst_index - 1])
            if (inst_labels[inst_index - 1] < 80 and cover_ratio < 0.1) or (
                    inst_labels[inst_index - 1] >= 80 and cover_ratio < 0.1):
                _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                pan_index[_mask] = inst_index
                if inst_labels[inst_index - 1] < 80:
                    pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                    id_unique += 1
                else:
                    pan_results[_mask] = inst_labels[inst_index - 1]
                area_list[inst_index] = np.sum(_mask)
                cover_flag = True
            elif rel_score > 0.1:
                _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                for k in range(len(cover_index_list)):
                    if cover_statistic[cover_index_list[k]] / area_list[cover_index_list[k]] < 0.1 and area_list[cover_index_list[k]] / np.sum(init_pan_masks[int(cover_index_list[k]) - 1]) > 0.6:
                        _mask |= (init_pan_masks[inst_index - 1] & (pan_index == cover_index_list[k]))
                        area_list[cover_index_list[k]] -= cover_statistic[cover_index_list[k]]
                area = np.sum(_mask)
                init_area = np.sum(init_pan_masks[inst_index - 1])
                #if ((inst_labels[inst_index - 1] < 80 and area > 256) or (
                #        inst_labels[inst_index - 1] >= 80 and area > 1024)) and area / init_area > 0.6:
                if area / init_area > 0.6:
                    pan_index[_mask] = inst_index
                    if inst_labels[inst_index - 1] < 80:
                        pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                        id_unique += 1
                    else:
                        pan_results[_mask] = inst_labels[inst_index - 1]
                    cover_flag = True
                    area_list[inst_index] = np.sum(_mask)
            # elif rel_score > 0.1:
            '''else:
                allowed_cover = True
                for k in range(len(cover_index_list)):
                    if cover_statistic[cover_index_list[k]] / area_list[cover_index_list[k]] > 0.1:
                        allowed_cover = False
                        break
                if allowed_cover:  # and np.sum(init_pan_masks[inst_index - 1]) > 4096:
                    # if (inst_labels[inst_index - 1] < 80) or (inst_labels[inst_index - 1] >= 80 and np.sum(init_pan_masks[inst_index - 1]) > 4096):
                    _mask = init_pan_masks[inst_index - 1]
                    pan_index[_mask] = inst_index
                    if inst_labels[inst_index - 1] < 80:
                        pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                        id_unique += 1
                    else:
                        pan_results[_mask] = inst_labels[inst_index - 1]
                    cover_flag = True
                    area_list[inst_index] = np.sum(_mask)
                    for k in range(len(cover_index_list)):
                        area_list[cover_index_list[k]] -= cover_statistic[cover_index_list[k]]
                allowed_cover = True
                for k in range(len(cover_index_list)):
                    if cover_statistic[cover_index_list[k]] / area_list[cover_index_list[k]] > 0.1:
                        allowed_cover = False
                        break
                if allowed_cover: #  and np.sum(init_pan_masks[inst_index - 1]) > 4096:
                # if (inst_labels[inst_index - 1] < 80) or (inst_labels[inst_index - 1] >= 80 and np.sum(init_pan_masks[inst_index - 1]) > 4096):
                    _mask = init_pan_masks[inst_index - 1]
                    pan_index[_mask] = inst_index
                    if inst_labels[inst_index - 1] < 80:
                        pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                        id_unique += 1
                    else:
                        pan_results[_mask] = inst_labels[inst_index - 1]
                    cover_flag = True
                    area_list[inst_index] = np.sum(_mask)
                    for k in range(len(cover_index_list)):
                        area_list[cover_index_list[k]] -= cover_statistic[cover_index_list[k]]'''

        return pan_index, pan_results, area_list, cover_flag, id_unique

    @staticmethod
    def get_covered_area(init_mask, cover_mask):
        cover_statistic = {}
        cover_area = 0

        rest_mask = init_mask * cover_mask
        index_list, index_counts = np.unique(rest_mask, return_counts=True)
        if index_list[0] == 0:
            index_list = index_list[1:]
            index_counts = index_counts[1:]
        for i in range(len(index_list)):
            cover_statistic[index_list[i]] = index_counts[i]
            cover_area += index_counts[i]

        return cover_statistic, index_list, cover_area
