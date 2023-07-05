import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.core import BitmapMasks, bbox2roi, build_assigner, multiclass_nms
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap

from ..detectors.panseg import PanSeg
from ..detectors.detr_plus import DETR_plus
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Mask2FormerPanoptic(MaskFormer):
    def __init__(
        self,
        backbone,
        neck=None,
        panoptic_head=None,
        panoptic_fusion_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        relation_head=None,
    ):
        super(Mask2FormerPanoptic, self).__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

        self.count = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.test_cfg is not None:
            self.use_pesuedo_inference = self.test_cfg.use_peseudo_inference

        if self.train_cfg is not None:
            self.use_full_train_samples = self.train_cfg.use_full_train_samples
            self.use_proposal_matching = self.train_cfg.use_proposal_matching

        # Init relation head
        if relation_head is not None:
            self.relation_head = build_head(relation_head)

        # Cache the detection results to speed up the sgdet training.
        self.det_results = dict()

    @property
    def with_relation(self):
        return hasattr(self,
                       'relation_head') and self.relation_head is not None

    def simple_test_sg_bboxes(self,
                              x,
                              img_metas,
                              proposals=None,
                              rescale=False,
                              peseudo_test=False,
                              peseudo_inference=False,
                              proposal_matching=False):
        """Test without Augmentation; convert panoptic segments to bounding
        boxes."""

        mask_cls_results, mask_pred_results, obj_query, feature_maps = self.panoptic_head.simple_test(
            x, img_metas)
        results = self.panoptic_fusion_head.simple_test(
            mask_cls_results, mask_pred_results, img_metas, peseudo_test)
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach(
                ).cpu().numpy()

            mask_pred_results = list(mask_pred_results)
            for i in range(len(results)):
                img_height, img_width = img_metas[i]['img_shape'][:2]
                mask_pred_results[i] = mask_pred_results[i][:, :img_height, :img_width]

        if peseudo_test:
            # return result in original resolution
            ori_height, ori_width = img_metas[0]['ori_shape'][:2]
            mask_pred_results[0] = F.interpolate(
                mask_pred_results[0].unsqueeze(0),
                size=(ori_height, ori_width),
                mode='nearest').squeeze(0)

        final_result = []

        for i in range(len(results)):
            mask_cls = mask_cls_results[i]
            mask_pred = mask_pred_results[i]
            obj_feats = obj_query[i]
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
            mask_pred = mask_pred.sigmoid()
            valid_idx = labels.ne(self.num_classes)
            use_idx = valid_idx.clone()

            '''seg = mask_pred > 0.5
            sum_seg = seg.sum((1, 2)).float() + 1
            seg_score = (mask_pred * seg.float()).sum((1, 2)) / sum_seg
            scores = seg_score * scores'''
            # temp_score = scores[valid_idx]

            valid_obj_feats = []
            pan_results = results[i]['pan_results']
            label2idx = results[i]['label2idx']
            used_idx = torch.tensor(results[i]['used_idx']).to(labels)
            use_idx[used_idx] = False
            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.num_classes
            ids = ids[legal_indices]

            for j in range(len(ids)):
                if len(label2idx[ids[j]]) == 1:
                    valid_obj_feats.append(obj_feats[label2idx[ids[j]][0]])
                else:
                    valid_obj_feats.append(sum(obj_feats[label2idx[ids[j]]]))
            if len(ids) == 0:
                print('\n zerp prediction!')
                valid_obj_feats = torch.tensor([]).to(obj_feats)
            else:
                valid_obj_feats = torch.stack(valid_obj_feats).to(obj_feats)


            final_labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64) + 1
            stuff_flag = final_labels >= self.num_things_classes
            segms = pan_results[None] == ids[:, None, None]

            height, width = segms.shape[1:]
            masks_to_bboxes = BitmapMasks(segms, height, width).get_bboxes()

            masks_to_bboxes = torch.tensor(masks_to_bboxes).to(scores)
            final_labels = torch.tensor(final_labels).to(labels)
            stuff_flag = torch.tensor(stuff_flag).to(labels)

            # test_valid_sum = sum(valid_idx)
            # test_used_sum = sum(use_idx)

            if (self.training and proposal_matching) or (not self.training and peseudo_inference):
                single_result = dict(pan_results=pan_results,
                                     masks=segms,
                                     bboxes=masks_to_bboxes,
                                     labels=final_labels,
                                     obj_query=valid_obj_feats,
                                     stuff_flag=stuff_flag,
                                     use_index=use_idx,
                                     rest_query=obj_feats[valid_idx],
                                     rest_bboxes=None,
                                     rest_masks=mask_pred[valid_idx],
                                     rest_labels=labels[valid_idx] + 1,
                                     rest_scores=scores[valid_idx],
                                     )
                '''single_result = dict(pan_results=None,
                                     masks=None,
                                     bboxes=None,
                                     labels=None,
                                     obj_query=valid_obj_feats,
                                     stuff_flag=None,
                                     use_index=None,
                                     rest_query=obj_feats[valid_idx],
                                     rest_bboxes=None,
                                     rest_masks=mask_pred[valid_idx],
                                     rest_labels=labels[valid_idx] + 1,
                                     rest_scores=scores[valid_idx],
                                     )'''
            else:
                single_result = dict(pan_results=pan_results,
                                     masks=segms,
                                     bboxes=masks_to_bboxes,
                                     labels=final_labels,
                                     obj_query=valid_obj_feats,
                                     stuff_flag=stuff_flag,
                                     use_index=use_idx,
                                     rest_query=obj_feats[use_idx],
                                     rest_bboxes=None,
                                     rest_masks=mask_pred[use_idx],
                                     rest_labels=labels[use_idx],
                                     )

            final_result.append(single_result)


        return final_result

    def set_epoch(self, epoch):
        self.backbone.epoch = epoch

    def set_max_epochs(self, max_epochs):
        self.backbone.max_epochs = max_epochs

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        gt_rels=None,
        gt_keyrels=None,
        gt_relmaps=None,
        gt_scenes=None,
        rescale=False,
        **kwargs,
    ):
        # img: (B, C, H, W)
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        x = self.extract_feat(img)

        if self.with_relation:
            gt_labels = [label + 1 for label in gt_labels]

            (
                bboxes,
                labels,
                target_labels,
                target_inds,
                target_masks,
                dists,  # Can this be `None`?
                pan_masks,
                pan_results,
                points,
                obj_feats_query,
                seg_scores,
            ) = self.detector_simple_test(
                x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_masks,
                proposals,
                use_gt_box=self.relation_head.use_gt_box,
                use_gt_label=self.relation_head.use_gt_label,
                rescale=rescale,
            )

            # Filter out empty predictions
            idxes_to_filter = [i for i, b in enumerate(bboxes) if len(b) == 0]

            param_need_filter = [
                bboxes, labels, dists, target_labels, target_inds, target_masks, gt_bboxes, gt_labels,
                gt_rels, img_metas, gt_scenes, gt_keyrels, points, pan_results,
                gt_masks, gt_relmaps, pan_masks, obj_feats_query
            ]
            for idx, param in enumerate(param_need_filter):
                if param_need_filter[idx]:
                    param_need_filter[idx] = [
                        x for i, x in enumerate(param)
                        if i not in idxes_to_filter
                    ]

            (bboxes, labels, dists, target_labels, target_inds, target_masks, gt_bboxes, gt_labels,
             gt_rels, img_metas, gt_scenes, gt_keyrels, points, pan_results,
             gt_masks, gt_relmaps, pan_masks, obj_feats_query) = param_need_filter
            # Filter done

            if idxes_to_filter and len(gt_bboxes) == 16:
                print('sg_panoptic_fpn: not filtered!')

            filtered_x = []
            for idx in range(len(x)):
                filtered_x.append(
                    torch.stack([
                        e for i, e in enumerate(x[idx])
                        if i not in idxes_to_filter
                    ]))
            x = filtered_x

            gt_result = Result(
                # bboxes=all_gt_bboxes,
                # labels=all_gt_labels,
                bboxes=gt_bboxes,
                labels=gt_labels,
                rels=gt_rels,
                relmaps=gt_relmaps,
                masks=gt_masks,
                rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels]
                if gt_rels is not None else None,
                rel_labels=[rel[:, -1].clone() for rel in gt_rels]
                if gt_rels is not None else None,
                key_rels=gt_keyrels if gt_keyrels is not None else None,
                img_shape=[meta['img_shape'] for meta in img_metas],
                scenes=gt_scenes,
            )

            det_result = Result(
                bboxes=bboxes,
                labels=labels,
                dists=dists,
                masks=pan_masks,
                pan_results=pan_results,
                points=points,
                target_labels=target_labels,
                target_inds=target_inds,
                target_masks=target_masks,
                target_scenes=gt_scenes,
                obj_feats_query=obj_feats_query,
                img_shape=[meta['img_shape'] for meta in img_metas],
            )

            det_result.epoch = self.backbone.epoch
            det_result.max_epochs = self.backbone.max_epochs


            det_result = self.relation_head(x, img_metas, det_result,
                                            gt_result)

            # Loss performed here
            return self.relation_head.loss(det_result)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        key_first = kwargs.pop('key_first', False)

        # print(imgs[0].shape)

        # if relation_mode:
        # assert num_augs == 1
        return self.relation_simple_test(imgs[0],
                                         img_metas[0],
                                         key_first=key_first,
                                         **kwargs)

        # if num_augs == 1:
        #     # proposals (List[List[Tensor]]): the outer list indicates
        #     # test-time augs (multiscale, flip, etc.) and the inner list
        #     # indicates images in a batch.
        #     # The Tensor should have a shape Px4, where P is the number of
        #     # proposals.
        #     if "proposals" in kwargs:
        #         kwargs["proposals"] = kwargs["proposals"][0]
        #     return self.simple_test(imgs[0], img_metas[0], **kwargs)
        # else:
        #     assert imgs[0].size(0) == 1, (
        #         "aug test does not support "
        #         "inference with batch size "
        #         f"{imgs[0].size(0)}"
        #     )
        #     # TODO: support test augmentation for predefined proposals
        #     assert "proposals" not in kwargs
        #     return self.aug_test(imgs, img_metas, **kwargs)

    def detector_simple_test(
        self,
        x,
        img_meta,
        gt_bboxes,
        gt_labels,
        gt_masks,
        proposals=None,
        use_gt_box=False,
        use_gt_label=False,
        rescale=False,
        is_testing=False,
    ):
        """Test without augmentation. Used in SGG.

        Return:
            det_bboxes: (list[Tensor]): The boxes may have 5 columns (sgdet)
                or 4 columns (predcls/sgcls).
            det_labels: (list[Tensor]): 1D tensor, det_labels (sgdet) or
                gt_labels (predcls/sgcls).
            det_dists: (list[Tensor]): 2D tensor, N x Nc, the bg column is 0.
                detected dists (sgdet/sgcls), or None (predcls).
            masks: (list[list[Tensor]]): Mask is associated with box. Thus,
                in predcls/sgcls mode, it will firstly return the gt_masks.
                But some datasets do not contain gt_masks. We try to use the
                gt box to obtain the masks.
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'

        pan_seg_masks = None

        if not is_testing:  # excluding the testing phase
            det_results = self.simple_test_sg_bboxes(x,
                                                     img_meta,
                                                     rescale=rescale,
                                                     peseudo_test=False,
                                                     proposal_matching=self.use_proposal_matching)
            det_bboxes = [r['bboxes'] for r in det_results]
            det_labels = [r['labels'] for r in det_results]  # 1-index
            pan_seg_masks = [torch.tensor(r['masks']).to(gt_bboxes[0]) for r in det_results]
            obj_feats_query = [r['obj_query'] for r in det_results]
            rest_obj_query = [r['rest_query'] for r in det_results]
            # rest_obj_bboxes = [r['rest_bboxes'] for r in det_results]
            rest_obj_masks = [r['rest_masks'] for r in det_results]
            rest_obj_labels = [r['rest_labels'] for r in det_results]
            # stuff_flag = [r['stuff_flag'] for r in det_results]
            pan_results = None
            seg_scores = None

            pred_cls = [
                F.one_hot(det_label - 1,
                          num_classes=self.num_classes).to(det_bboxes[0])
                for det_label in det_labels
            ]

            target_labels = []
            target_inds = []
            target_masks = []
            # bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            mask_assigner = build_assigner(self.train_cfg.mask.assigner)
            for i in range(len(img_meta)):
                if self.use_proposal_matching:
                    unused_flag = torch.ones(len(gt_labels[i])).bool()
                    map2init = torch.nonzero(unused_flag).view(-1).to(det_labels[-1])
                    pred_cls = F.one_hot(rest_obj_labels[i] - 1,
                                              num_classes=self.num_classes).to(det_bboxes[0])
                    mask_assign_result = mask_assigner.assign(
                        pred_cls,
                        rest_obj_masks[i],
                        gt_labels[i] - 1,
                        torch.tensor(gt_masks[i].masks).to(gt_bboxes[0]),
                        img_meta[i],
                    )
                    choose_index = mask_assign_result.labels >= 0
                    target_labels.append(mask_assign_result.labels[choose_index] + 1)
                    temp_inds = map2init[mask_assign_result.gt_inds[choose_index] - 1] + 1
                    target_inds.append(temp_inds)
                    target_masks.append(torch.from_numpy(gt_masks[i].masks)[temp_inds - 1])
                    det_labels[i] = rest_obj_labels[i][choose_index]
                    temp_height, temp_width = rest_obj_masks[i].shape[1:]
                    rest_obj_masks[i] = rest_obj_masks[i] >= 0.5
                    masks_to_bboxes = BitmapMasks(rest_obj_masks[i][choose_index].detach().cpu().numpy(), temp_height,
                                                  temp_width).get_bboxes()
                    masks_to_bboxes = torch.tensor(masks_to_bboxes).to(det_bboxes[i])
                    det_bboxes[i] = masks_to_bboxes
                    # det_bboxes[i] = torch.cat((det_bboxes[i], rest_obj_bboxes[i][choose_index]))
                    obj_feats_query[i] = rest_obj_query[i][choose_index]

                else:
                    mask_assign_result = mask_assigner.assign(
                        pred_cls[i],
                        pan_seg_masks[i],
                        gt_labels[i] - 1,
                        torch.tensor(gt_masks[i].masks).to(gt_bboxes[0]),
                        img_meta[i],
                    )
                    target_labels.append(mask_assign_result.labels + 1)
                    target_inds.append(mask_assign_result.gt_inds)
                    target_masks.append(torch.from_numpy(gt_masks[i].masks)[target_inds[-1] - 1])

                    if self.use_full_train_samples and len(pred_cls[i]) < len(gt_labels[i]):
                        rest_mask_assigner = build_assigner(self.train_cfg.rest_mask.assigner)
                        unused_flag = torch.ones(len(gt_labels[i])).bool()
                        unused_flag[mask_assign_result.gt_inds - 1] = False
                        map2init = torch.nonzero(unused_flag).view(-1).to(target_inds[-1])
                        rest_pred_cls = F.one_hot(rest_obj_labels[i],
                                      num_classes=self.num_classes).to(det_bboxes[0])
                        rest_mask_assign_result = rest_mask_assigner.assign(
                            rest_pred_cls,
                            rest_obj_masks[i],
                            gt_labels[i][unused_flag] - 1,
                            torch.tensor(gt_masks[i].masks[unused_flag]).to(gt_bboxes[0]),
                            img_meta[i],
                        )
                        choose_index = rest_mask_assign_result.labels >= 0
                        target_labels[-1] = torch.cat((target_labels[-1], rest_mask_assign_result.labels[choose_index] + 1))
                        target_inds[-1] = torch.cat((target_inds[-1], map2init[rest_mask_assign_result.gt_inds[choose_index] - 1] + 1))
                        target_masks[-1] = torch.from_numpy(gt_masks[i].masks)[target_inds[-1] - 1]
                        det_labels[i] = torch.cat((det_labels[i], rest_obj_labels[i][choose_index] + 1))
                        temp_height, temp_width = rest_obj_masks[i].shape[1:]
                        rest_obj_masks[i] = rest_obj_masks[i] > 0.5
                        masks_to_bboxes = BitmapMasks(rest_obj_masks[i][choose_index].detach().cpu().numpy(), temp_height, temp_width).get_bboxes()
                        masks_to_bboxes = torch.tensor(masks_to_bboxes).to(det_bboxes[i])
                        det_bboxes[i] = torch.cat((det_bboxes[i], masks_to_bboxes))
                        # det_bboxes[i] = torch.cat((det_bboxes[i], rest_obj_bboxes[i][choose_index]))
                        obj_feats_query[i] = torch.cat((obj_feats_query[i], rest_obj_query[i][choose_index]))


                '''bbox_assign_result = bbox_assigner.assign(
                    det_bboxes[i],
                    gt_bboxes[i],
                    gt_labels=gt_labels[i] - 1,
                )'''
                # target_labels.append(bbox_assign_result.labels + 1)
        else:
            det_results = self.simple_test_sg_bboxes(x,
                                                     img_meta,
                                                     rescale=rescale,
                                                     peseudo_test=True,
                                                     peseudo_inference=self.use_pesuedo_inference)
            if not self.use_pesuedo_inference:
                det_bboxes = [r['bboxes'] for r in det_results]
                det_labels = [r['labels'] for r in det_results]  # 1-index
                pan_seg_masks = [r['masks'] for r in det_results]
                pan_seg_masks = [torch.Tensor(pan_seg_masks[0]).bool().numpy()]
                obj_feats_query = [r['obj_query'] for r in det_results]
                pan_results = [r['pan_results'] for r in det_results]
                target_labels = None
                target_masks = None
                target_inds = None
            else:
                peseudo_inference_threshold = 0.2
                num_proposal = [sum(r['rest_scores'] > peseudo_inference_threshold) for r in det_results]
                seg_scores = [det_results[i]['rest_scores'][:num_proposal[i]] for i in range(len(det_results))]
                # det_bboxes = [det_results[i]['rest_bboxes'][:num_proposal[i]] for i in range(len(det_results))]
                det_labels = [det_results[i]['rest_labels'][:num_proposal[i]] for i in range(len(det_results))]
                pan_seg_masks = [det_results[i]['rest_masks'][:num_proposal[i]] for i in range(len(det_results))]
                pan_seg_masks = [(pan_seg_masks[0] > 0.5).cpu().numpy()]
                temp_height, temp_width = pan_seg_masks[0].shape[1:]
                det_bboxes = [torch.tensor(BitmapMasks(pan_seg_masks[0], temp_height, temp_width).get_bboxes()).to(seg_scores[0])]
                obj_feats_query = [det_results[i]['rest_query'][:num_proposal[i]] for i in range(len(det_results))]

                pan_results = [r['pan_results'] for r in det_results]
                target_labels = None
                target_inds = None
                target_masks = None
            temp_bboxes = det_bboxes[0]
            temp_scale = torch.tensor(img_meta[0]['scale_factor']).to(temp_bboxes)
            temp_bboxes = torch.floor(temp_bboxes * temp_scale)
            det_bboxes[0] = temp_bboxes

        det_dists = [
            F.one_hot(det_label,
                      num_classes=self.num_classes + 1).to(det_bboxes[0])
            for det_label in det_labels
        ]

        det_bboxes = [
            torch.cat([b, b.new_ones(len(b), 1)], dim=-1)
            for b in det_bboxes
        ]

        if self.use_pesuedo_inference:
            return det_bboxes, det_labels, target_labels, target_inds, target_masks, \
                   det_dists, pan_seg_masks, pan_results, None, obj_feats_query, seg_scores
        else:
            return det_bboxes, det_labels, target_labels, target_inds, target_masks, \
                det_dists, pan_seg_masks, pan_results, None, obj_feats_query, None

    def relation_simple_test(
        self,
        img,
        img_meta,
        # all_gt_bboxes=None,
        # all_gt_labels=None,
        gt_bboxes=None,
        gt_labels=None,
        gt_rels=None,
        gt_masks=None,
        gt_scenes=None,
        rescale=False,
        ignore_classes=None,
        key_first=False,
    ):
        """
        :param img:
        :param img_meta:
        :param gt_bboxes: Usually, under the forward (train/val/test),
        it should not be None. But when for demo (inference), it should
        be None. The same for gt_labels.
        :param gt_labels:
        :param gt_rels: You should make sure that the gt_rels should not
        be passed into the forward process in any mode. It is only used to
        visualize the results.
        :param gt_masks:
        :param rescale:
        :param ignore_classes: For practice, you may want to ignore some
        object classes
        :return:
        """
        # Extract the outer list: Since the aug test is
        # temporarily not supported.
        # if all_gt_bboxes is not None:
        #     all_gt_bboxes = all_gt_bboxes[0]
        # if all_gt_labels is not None:
        #     all_gt_labels = all_gt_labels[0]
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]
        if gt_masks is not None:
            gt_masks = gt_masks[0]

        x = self.extract_feat(img)
        """
        NOTE: (for VG) When the gt masks is None, but the head needs mask,
        we use the gt_box and gt_label (if needed) to generate the fake mask.
        """

        # NOTE: Change to 1-index here:
        if gt_labels is not None:
            gt_labels = [label + 1 for label in gt_labels]

        # Rescale should be forbidden here since the bboxes and masks will
        # be used in relation module.
        bboxes, labels, target_labels, target_inds, target_masks, dists, pan_masks, pan_results, points, obj_feats_query, seg_scores \
            = self.detector_simple_test(
            x,
            img_meta,
            gt_bboxes,
            gt_labels,
            gt_masks,
            use_gt_box=self.relation_head.use_gt_box,
            use_gt_label=self.relation_head.use_gt_label,
            rescale=False,
            is_testing=True,
        )

        det_result = Result(
            bboxes=bboxes,
            labels=labels,
            dists=dists,
            masks=pan_masks,
            pan_results=pan_results,
            points=points,
            target_labels=target_labels,
            obj_feats_query=obj_feats_query,
            # saliency_maps=saliency_maps,
            img_shape=[meta['img_shape'] for meta in img_meta],
        )

        # If empty prediction
        if len(bboxes[0]) == 0:
            det_result.pan_results = det_result.pan_results[0]
            return det_result

        det_result.epoch = None
        det_result.max_epochs = None

        det_result = self.relation_head(x,
                                        img_meta,
                                        det_result,
                                        is_testing=True,
                                        ignore_classes=ignore_classes)
        """
        Transform the data type, and rescale the bboxes and masks if needed
        (for visual, do not rescale, for evaluation, rescale).
        """
        scale_factor = img_meta[0]['scale_factor']
        det_result = self.relation_head.get_result(det_result,
                                                   scale_factor,
                                                   rescale=False,
                                                   key_first=key_first)

        if pan_masks is not None:
            det_result.masks = np.array(pan_masks[0])

        if self.use_pesuedo_inference:
            det_result.seg_scores = seg_scores
            pan_results, pan_masks, pair_idxes, inst_idxes, idxes_map = self.backward_panoptic_inference(det_result)
            # diff = pan_results == det_result.pan_results
            # pan_results_copy = pan_results.copy()
            # det_pan_results_copy = det_result.pan_results.copy()
            # diff_graph = pan_results[diff]
            # diff_statistic, counts = np.unique(diff, return_counts=True)
            det_result.pan_results = pan_results
            det_result.masks = pan_masks
            # area_list = [np.sum(item) for item in pan_masks]
            # masks, refine_bboxes, refine_labels, rel_dists, rel_pair_idxes, rels
            det_result.refine_bboxes = det_result.refine_bboxes[inst_idxes, :]
            det_result.refine_dists = det_result.refine_dists[inst_idxes, :]
            det_result.refine_scores = det_result.refine_scores[inst_idxes, :]
            # det_result.refine_labels = det_result.refine_labels[inst_idxes]
            det_result.refine_labels = det_result.labels[inst_idxes]
            det_result.rel_dists = det_result.rel_dists[pair_idxes, :]
            det_result.rel_pair_idxes = idxes_map[det_result.rel_pair_idxes[pair_idxes, :]]
            # if np.max(det_result.rel_pair_idxes) >= det_result.refine_labels.shape[0]:
            #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            det_result.rels = det_result.rels[pair_idxes, :]
            det_result.rels[:, :2] = det_result.rel_pair_idxes
            # print(np.max(det_result.rel_pair_idxes))

        return det_result

    def backward_panoptic_inference(self, det_result):
        seg_scores = det_result.seg_scores[0]
        init_pan_masks = det_result.masks
        refine_obj_labels = det_result.labels  - 1
        refine_obj_scores = det_result.refine_scores
        rel_scores = det_result.rel_scores
        # rel_class_prob = F.softmax(rel_scores, -1)
        # rel_scores, _ = rel_class_prob[:, 1:].max(dim=1)
        triplet_scores = det_result.triplet_scores
        relations = det_result.rels

        # seg_pair_scores =
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
                '''if refine_obj_labels[rel[0]] < 80:
                    idx2score[rel[0]] += 0.1
                    idx2relscore[rel[0]] += 0.1'''
                used_flag[rel[0]] = 1
            if used_flag[rel[1]] == 0:
                temp_score = triplet_scores[i] * seg_scores[rel[1]]
                idx2score[rel[1]] = temp_score
                idx2relscore[rel[1]] = triplet_scores[i]
                '''if refine_obj_labels[rel[0]] < 80:
                    idx2score[rel[0]] += 0.1
                    idx2relscore[rel[1]] += 0.1'''
                used_flag[rel[1]] = 1

        # idx2score =

        sorted_score = sorted(idx2score.items(), key=lambda x: x[1], reverse=True)

        # sorted_score_1 = np.array([x[1].cpu() for x in sorted_score])
        # sorted_score_1_num_1 = np.sum(seg_scores.cpu().numpy() > 0.5)
        # sorted_score_1_num_2 = np.sum(seg_scores.cpu().numpy() > 0.8)
        # instance_cover_sequence = [item[0] + 1 for item in sorted_score if item[1] > 0.04]
        # instance_cover_sequence = [item[0] + 1 for item in sorted_score]
        instance_cover_sequence = [item[0] + 1 for item in sorted_score if seg_scores[item[0]] > 0.7]

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
        '''instance_cover_num = np.sum(seg_scores.cpu().numpy() > 0.7)
        sorted_seg_scores, sorted_idxes = torch.sort(seg_scores, descending=True)
        instance_cover_sequence = sorted_idxes[:instance_cover_num].cpu().numpy() + 1'''
        # instance_cover_sequence = sorted_idxes.cpu().numpy() + 1
        label_set = []
        old2map = {}

        for inst_index in instance_cover_sequence:
            # special for universal panoptic segmentation framework
            '''if refine_obj_labels[inst_index - 1] >= 80 and (refine_obj_labels[inst_index - 1] in sorted_instance_labels):
                old_index = sorted_instance_labels.index(refine_obj_labels[inst_index - 1])
                _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                pan_index[_mask] = sorted_instance_index[old_index]
                pan_results[_mask] = refine_obj_labels[inst_index - 1]
                old2map[inst_index - 1] = sorted_instance_index[old_index] - 1
                continue'''
            pan_index, pan_results, area_list, cover_flag, id_unique = self.get_cover_mask(pan_index, pan_results,
                                                                                           init_pan_masks, area_list,
                                                                                           inst_index, refine_obj_labels, idx2relscore[inst_index - 1],
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
            if len(old2map) > 0:
                for key, item in old2map.items():
                    idxes_map[key] = idxes_map[item]

        used_index = np.zeros(init_pan_masks.shape[0], dtype=np.bool)
        if len(sorted_instance_index) != 0:
            used_index[sorted_instance_index] = True
            if len(old2map) > 0:
                for key, item in old2map.items():
                    used_index[key] = True

        pan_masks = pan_results[None] == sorted_instance_labels[:, None, None]

        sub_used_list, obj_used_list = used_index[relations[:, 0]], used_index[relations[:, 1]]
        pair_used_idxes = sub_used_list & obj_used_list

        return pan_results, pan_masks, pair_used_idxes, sorted_instance_index, idxes_map

    def get_cover_mask(self, pan_index, pan_results, init_pan_masks, area_list, inst_index, inst_labels, rel_score, id_unique):

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
            if (inst_labels[inst_index - 1] < 80 and cover_ratio < 0.1) or (inst_labels[inst_index - 1] >= 80 and cover_ratio < 0.1):
                _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                pan_index[_mask] = inst_index
                if inst_labels[inst_index - 1] < 80:
                    pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                    id_unique += 1
                else:
                    pan_results[_mask] = inst_labels[inst_index - 1]
                area_list[inst_index] = np.sum(_mask)
                cover_flag = True
            elif rel_score > 2.0:
                allowed_cover = True
                for k in range(len(cover_index_list)):
                    if cover_statistic[cover_index_list[k]] / area_list[cover_index_list[k]] > 0.2:
                        allowed_cover = False
                        break
                if allowed_cover  and (inst_labels[inst_index - 1] < 80 or (inst_labels[inst_index - 1] >= 80 and np.sum(init_pan_masks[inst_index - 1]) > 1024)):
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
            elif rel_score > 2.0:
                if (inst_labels[inst_index - 1] < 80 or (
                        inst_labels[inst_index - 1] >= 80 and np.sum(init_pan_masks[inst_index - 1]) > 4096)):
                    _mask = init_pan_masks[inst_index - 1] & (pan_index == 0)
                    for k in range(len(cover_index_list)):
                        if cover_statistic[cover_index_list[k]] / area_list[cover_index_list[k]] < 0.2:
                            _mask |= (init_pan_masks[inst_index - 1] & (pan_index == cover_index_list[k]))
                            area_list[cover_index_list[k]] -= cover_statistic[cover_index_list[k]]
                    pan_index[_mask] = inst_index
                    if inst_labels[inst_index - 1] < 80:
                        pan_results[_mask] = inst_labels[inst_index - 1] + id_unique * INSTANCE_OFFSET
                        id_unique += 1
                    else:
                        pan_results[_mask] = inst_labels[inst_index - 1]
                    cover_flag = True
                    area_list[inst_index] = np.sum(_mask)
            elif rel_score > 0.01: # 0.00005
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

    def show_result(
        self,
        img,
        result,
        score_thr=0.3,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        mask_color=None,
        thickness=2,
        font_size=13,
        win_name='',
        show=False,
        wait_time=0,
        out_file=None,
    ):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        self.CLASSES = [
            'airplane',
            'apple',
            'backpack',
            'banana',
            'baseball bat',
            'baseball glove',
            'bear',
            'bed',
            'bench',
            'bicycle',
            'bird',
            'boat',
            'book',
            'bottle',
            'bowl',
            'broccoli',
            'bus',
            'cake',
            'car',
            'carrot',
            'cat',
            'cell phone',
            'chair',
            'clock',
            'couch',
            'cow',
            'cup',
            'dining table',
            'dog',
            'donut',
            'elephant',
            'fire hydrant',
            'fork',
            'frisbee',
            'giraffe',
            'hair drier',
            'handbag',
            'horse',
            'hot dog',
            'keyboard',
            'kite',
            'knife',
            'laptop',
            'microwave',
            'motorcycle',
            'mouse',
            'orange',
            'oven',
            'parking meter',
            'person',
            'pizza',
            'potted plant',
            'refrigerator',
            'remote',
            'sandwich',
            'scissors',
            'sheep',
            'sink',
            'skateboard',
            'skis',
            'snowboard',
            'spoon',
            'sports ball',
            'stop sign',
            'suitcase',
            'surfboard',
            'teddy bear',
            'tennis racket',
            'tie',
            'toaster',
            'toilet',
            'toothbrush',
            'traffic light',
            'train',
            'truck',
            'tv',
            'umbrella',
            'vase',
            'wine glass',
            'zebra',
            'banner',
            'blanket',
            'bridge',
            'building',
            'cabinet',
            'cardboard',
            'ceiling',
            'counter',
            'curtain',
            'dirt',
            'door',
            'fence',
            'floor',
            'floor-wood',
            'flower',
            'food',
            'fruit',
            'grass',
            'gravel',
            'house',
            'light',
            'mirror',
            'mountain',
            'net',
            'paper',
            'pavement',
            'pillow',
            'platform',
            'playingfield',
            'railroad',
            'river',
            'road',
            'rock',
            'roof',
            'rug',
            'sand',
            'sea',
            'shelf',
            'sky',
            'snow',
            'stairs',
            'table',
            'tent',
            'towel',
            'tree',
            'wall-brick',
            'wall',
            'wall-stone',
            'wall-tile',
            'wall-wood',
            'water',
            'window-blind',
            'window',
        ]

        # Load image
        img = mmcv.imread(img)
        img = img.copy()  # (H, W, 3)
        img_h, img_w = img.shape[:-1]

        if True:
            # Draw masks
            pan_results = result.pan_results

            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.num_classes  # for VOID label
            ids = ids[legal_indices]

            # Get predicted labels
            labels = np.array([id % INSTANCE_OFFSET for id in ids],
                              dtype=np.int64)
            labels = [self.CLASSES[label] for label in labels]

            # (N_m, H, W)
            segms = pan_results[None] == ids[:, None, None]
            # Resize predicted masks
            segms = [
                mmcv.image.imresize(m.astype(float), (img_w, img_h))
                for m in segms
            ]

            # Choose colors for each instance in coco
            colormap_coco = get_colormap(len(segms))
            colormap_coco = (np.array(colormap_coco) / 255).tolist()

            viz = Visualizer(img)
            viz.overlay_instances(
                labels=labels,
                masks=segms,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()

        else:
            # Draw bboxes
            bboxes = result.refine_bboxes[:, :4]

            # Choose colors for each instance in coco
            colormap_coco = get_colormap(len(bboxes))
            colormap_coco = (np.array(colormap_coco) / 255).tolist()

            # 1-index
            labels = [self.CLASSES[label - 1] for label in result.labels]

            viz = Visualizer(img)
            viz.overlay_instances(
                labels=labels,
                boxes=bboxes,
                assigned_colors=colormap_coco,
            )
            viz_img = viz.get_output().get_image()

        # Draw relations

        # Filter out relations
        n_rel_topk = 20

        # Exclude background class
        rel_dists = result.rel_dists[:, 1:]
        # rel_dists = result.rel_dists

        rel_scores = rel_dists.max(1)
        # rel_scores = result.triplet_scores

        # Extract relations with top scores
        rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
        rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
        rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
        relations = np.concatenate(
            [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)

        n_rels = len(relations)

        top_padding = 20
        bottom_padding = 20
        left_padding = 20

        text_size = 10
        text_padding = 5
        text_height = text_size + 2 * text_padding

        row_padding = 10

        height = (top_padding + bottom_padding + n_rels *
                  (text_height + row_padding) - row_padding)
        width = viz_img.shape[1]

        curr_x = left_padding
        curr_y = top_padding

        # Adjust colormaps
        colormap_coco = [adjust_text_color(c, viz) for c in colormap_coco]

        viz_graph = VisImage(np.full((height, width, 3), 255))

        for i, r in enumerate(relations):
            s_idx, o_idx, rel_id = r

            s_label = labels[s_idx]
            o_label = labels[o_idx]
            # Becomes 0-index
            rel_label = self.PREDICATES[rel_id]

            # Draw subject text
            text_width = draw_text(
                viz_img=viz_graph,
                text=s_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[s_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            # Draw relation text
            text_width = draw_text(
                viz_img=viz_graph,
                text=rel_label,
                x=curr_x,
                y=curr_y,
                size=text_size,
                padding=text_padding,
                box_color='gainsboro',
                # font=font,
            )
            curr_x += text_width

            # Draw object text
            text_width = draw_text(
                viz_img=viz_graph,
                text=o_label,
                x=curr_x,
                y=curr_y,
                color=colormap_coco[o_idx],
                size=text_size,
                padding=text_padding,
                # font=font,
            )
            curr_x += text_width

            curr_x = left_padding
            curr_y += text_height + row_padding

        viz_graph = viz_graph.get_image()

        viz_final = np.vstack([viz_img, viz_graph])

        if out_file is not None:
            mmcv.imwrite(viz_final, out_file)

        if not (show or out_file):
            return viz_final
