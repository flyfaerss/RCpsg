from mmdet.core import bbox2result
# from ..builder import DETECTORS
from ..detectors.single_stage_panoptic_detector import SingleStagePanopticDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import torch
import torch.nn.functional as F
import numpy as np
from openpsg.models.relation_heads.approaches import Result
from mmcv.runner.fp16_utils import auto_fp16
from torch.utils.checkpoint import checkpoint
# from easymd.models.utils.transform import mask2result
from mmdet.core import bbox2result, bbox_mapping_back
import mmcv
from torchvision.transforms.transforms import ToTensor


def mask2result(seg, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """

    if seg.shape[0] == 0:
        _, h, w = seg.shape
        return [np.zeros((0, h, w), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [seg[labels == i, :] for i in range(num_classes)]


def triplet2Result(triplets, use_mask, eval_pan_rels=True):
    if use_mask:
        bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
            r_labels, r_dists, pan_masks, rels, pan_labels \
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
                        labels=pan_labels+1,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=pan_rel_pairs,# elif not pan: rel_pairs,
                        rel_dists=r_dists,
                        rel_labels=r_labels,
                        pan_results=pan_seg,
                        masks=pan_masks,
                        rels=rels)
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
class RelPanFormer(SingleStagePanopticDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        super(RelPanFormer, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        self.count = 0
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes
        self.num_things_classes = self.bbox_head.num_things_classes
        self.num_stuff_classes = self.bbox_head.num_stuff_classes

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """

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
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      gt_semantic_seg=None
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # mmcv.imshow(gt_semantic_seg.squeeze(0).squeeze(0).cpu().numpy())
        # mmcv.imshow(img.squeeze(0).permute(1,2,0).cpu().numpy())

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        # img_metas[0]['img'] = img
        # super(SingleStagePanopticDetector, self).forward_train(img, img_metas)
        if self.with_checkpoint:
            img.requires_grad_(True)
            x = checkpoint(self.extract_feat, img)
        else:
            x = self.extract_feat(img)
        BS, C, H, W = img.shape
        new_gt_masks = []
        for each in gt_masks:
            mask = torch.tensor(each.to_ndarray(), device=x[0].device)
            _, h, w = mask.shape
            padding = (
                0, W - w,
                0, H - h
            )
            mask = F.pad(mask, padding)
            new_gt_masks.append(mask)
        gt_masks = new_gt_masks

        losses = self.bbox_head.forward_train(x, img_metas, gt_rels, gt_bboxes,
                                              gt_labels, gt_masks, gt_bboxes_ignore, gt_semantic_seg=gt_semantic_seg)
        return losses

    def simple_test(self, img, img_metas=None, rescale=False):

        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
                                f'mode is supported. Found batch_size {batch_size}.'
        x = self.extract_feat(img)

        # outs = self.bbox_head(x, img_metas)
        # pan_results, results = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        results_list = self.bbox_head.simple_test(x,
                                                  img_metas,
                                                  rescale=rescale)
        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)
        return sg_results
