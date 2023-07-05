import mmcv
import numpy as np
import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.core import BitmapMasks, bbox2roi, build_assigner, multiclass_nms
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap

from ..detectors.panseg import PanSeg
from ..detectors.detr_plus import DETR_plus
from .diffusion_condition_model import DiffusionConditionModel
from .get_prototype import obj_edge_vectors
from torch.nn.functional import cosine_similarity


@DETECTORS.register_module()
class DiffusionSceneGraph(BaseDetector):
    def __init__(self,
                 diffusion_cfg,
                 panoptic_process_cfg,
                 condition_model_cfg,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 relation_head=None,
                 ):
        super(DiffusionSceneGraph, self).__init__(
            init_cfg=init_cfg,
        )

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        if relation_head is not None:
            self.relation_head = build_head(relation_head)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.diffusion_cfg = diffusion_cfg
        self.panoptic_process_cfg = panoptic_process_cfg

        self.inst_classes = self.diffusion_cfg.object_classes
        self.rel_classes = self.diffusion_cfg.predicate_classes
        self.inst_classes.insert(0, 'background')
        self.rel_classes.insert(0, 'background')

        self.inst_prototype = obj_edge_vectors(self.inst_classes, wv_dir=self.diffusion_cfg.glove_dir,
                                               wv_dim=self.diffusion_cfg.prototype_dim)
        self.rel_prototype = obj_edge_vectors(self.rel_classes, wv_dir=self.diffusion_cfg.glove_dir,
                                              wv_dim=self.diffusion_cfg.prototype_dim)

        if self.diffusion_cfg.proposal_refine:
            # todo: finish proposal refinement, a transformer layer is needed to refine this
            self.inst_cls_embed = nn.Linear(self.diffusion_cfg.input_dim, len(self.inst_classes))
            nn.init.xavier_normal_(self.inst_cls_embed)
            self.inst_cls_embed.weight.data.copy_(self.panoptic_head.cls_embed.weight.data)
            self.inst_cls_embed.bias.data.copy_(self.panoptic_head.cls_embed.bias.data)

        self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
        if self.diffusion_cfg.beta_fixed:
            self.betas[0] = 0.0001
        self.calculate_for_diffusion()
        self.condition_model_cfg = condition_model_cfg
        self.pred_noise_model = DiffusionConditionModel(self.condition_model_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def simple_test(self, img, img_metas, **kwargs):
        pass

    def set_epoch(self, epoch):
        self.backbone.epoch = epoch

    def set_max_epochs(self, max_epochs):
        self.backbone.max_epochs = max_epochs

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.diffusion_cfg.noise_schedule == "linear" or self.diffusion_cfg.noise_schedule == "linear-var":
            start = self.diffusion_cfg.noise_scale * self.diffusion_cfg.noise_min
            end = self.diffusion_cfg.noise_scale * self.diffusion_cfg.noise_max
            if self.diffusion_cfg.noise_schedule == "linear":
                return np.linspace(start, end, self.diffusion_cfg.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.diffusion_cfg.steps,
                                                  np.linspace(start, end, self.diffusion_cfg.steps, dtype=np.float64))
        elif self.diffusion_cfg.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.diffusion_cfg.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.diffusion_cfg.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.diffusion_cfg.steps)
            betas = [1 / (self.diffusion_cfg.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.diffusion_cfg.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])])  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.diffusion_cfg.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        '''else:
            self.posterior_mean_coef1 = (
                    self.betas / (torch.sqrt(1 - self.alphas_cumprod) * torch.sqrt(alphas))
            )
            self.posterior_mean_coef2 = (
                    1.0 / (torch.sqrt(alphas))
            )'''

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    '''def _extract_into_tensor(self, arr, timesteps, shape_list):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        len_expand = len(shape_list[0])
        for i in range(len_expand):
            res = res[..., None]
        results = [res[i].expand(shape_item) for i, shape_item in enumerate(shape_list)]

        return results'''

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        '''shape_list = [x_start[i].size() for i in range(len(x_start))]
        results = [self._extract_into_tensor(self.sqrt_alphas_cumprod, t, shape_list)[i] * x_start[i]
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, shape_list)[i]
            * noise[i] for i in range(len(x_start))]
        return results'''
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.diffusion_cfg.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.diffusion_cfg.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt
        else:
            raise ValueError

    def process_panoptic_head(self,
                              x,
                              img_metas,
                              proposals=None,
                              rescale=False):
        mask_cls_results, mask_pred_results, inst_features, feature_maps = self.panoptic_head.simple_test(x, img_metas)

        results = []
        for mask_cls_result, mask_pred_result, inst_feature, feature_map, img_meta in zip(
                mask_cls_results, mask_pred_results, inst_features, feature_maps, img_metas):
            img_height, img_width = img_meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = img_meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            mask_cls_score, mask_cls_label = F.softmax(mask_cls_result, dim=-1).max(-1)
            if self.panoptic_process_cfg.mask_sigmoid:
                mask_pred_result = mask_pred_result.sigmoid()

            keep = mask_cls_label.ne(self.num_classes) & (mask_cls_score > self.panoptic_process_cfg.mask_cls_threshold)
            proposal_feature = inst_feature[keep]
            proposal_cls = mask_cls_label[keep]
            proposal_cls_pred = mask_cls_result[keep]
            proposal_mask_pred = mask_pred_result[keep]
            proposal_mask = proposal_mask_pred >= self.panoptic_process_cfg.mask_threshold

            seg_result_final, seg_result_split = None, None
            # todo: need to complete segment function
            if self.panoptic_process_cfg.need_seg_results:
                seg_result_final = None
                seg_result_split = None

            result = dict(proposal_feature=proposal_feature,
                          proposal_cls=proposal_cls,
                          proposal_cls_pred=proposal_cls_pred,
                          proposal_mask=proposal_mask,
                          proposal_mask_pred=proposal_mask_pred,
                          feature_map=feature_map,
                          seg_result_final=seg_result_final,
                          seg_result_split=seg_result_split)
            results.append(result)
        return results

    def process_segment(self,
                              x,
                              img_metas,
                              gt_bboxes,
                              gt_labels,
                              gt_masks,
                              proposals=None,
                              use_gt_box=False,
                              use_gt_label=False,
                              rescale=False,
                              is_testing=False,
    ):
        det_results = self.process_panoptic_head(x,
                                             img_metas,
                                             rescale=rescale)
        proposal_features = [r['proposal_feature'] for r in det_results]
        proposal_clses = [r['proposal_cls'] for r in det_results]
        proposal_cls_preds = [r['proposal_cls_pred'] for r in det_results]
        proposal_masks = [r['proposal_mask'] for r in det_results]
        proposal_mask_preds = [r['proposal_mask_pred'] for r in det_results]
        proposals = dict(proposal_features=proposal_features,
                         proposal_clses=proposal_clses,
                         proposal_cls_preds=proposal_cls_preds,
                         proposal_masks=proposal_masks,
                         proposal_mask_preds=proposal_mask_preds)
        feature_maps = [r['feature_map'] for r in det_results]
        seg_result_finals = [r['seg_result_final'] for r in det_results]
        seg_result_splits = [r['seg_result_split'] for r in det_results]
        seg_results = dict(seg_result_finals=seg_result_finals,
                          seg_result_splits=seg_result_splits)

        return proposals, feature_maps, seg_results

    def forward_train(self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore = None,
        gt_masks = None,
        proposals = None,
        gt_rels = None,
        gt_keyrels = None,
        gt_relmaps = None,
        gt_scenes = None,
        rescale = False,
        **kwargs,
    ):

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        gt_labels = [label + 1 for label in gt_labels]

        x = self.extract_feat(img)
        (proposals, feature_maps, seg_results) = self.process_segment(x,
                                                                   img_metas,
                                                                   gt_bboxes,
                                                                   gt_labels,
                                                                   gt_masks,
                                                                   proposals,
                                                                   use_gt_box=self.diffusion_cfg.use_gt_box,
                                                                   use_gt_label=self.diffusion_cfg.use_gt_label,
                                                                   rescale=rescale)

        feature_maps = x if self.diffusion_cfg.multiscale_feature_map else feature_maps

        sgs_start, temp_proposals, gt_clses, inst_num, _, positive_samples = self.get_gtscenegraph(gt_labels, gt_rels, gt_masks, proposals, img_metas)
        # for debug
        # sgs_start_numpy = [sgs_start[i].cpu().numpy() for i in range(len(sgs_start))]

        proposals = proposals if self.diffusion_cfg.proposal_assign_gt else temp_proposals

        # in very special situation, the panoptic head may conduct zero proposal
        index_0 = [i for i, (item_1, item_2) in enumerate(zip(inst_num, positive_samples)) if
                   (item_1 == 0 or item_2.size(0) == 0)]
        if len(index_0) > 0:
            index_0_reverse = [i for i, (item_1, item_2) in enumerate(zip(inst_num, positive_samples)) if
                               (item_1 != 0 and item_2.size(0) != 0)]
            for index in sorted(index_0, reverse=True):
                del inst_num[index], feature_maps[index], gt_clses[index], img_metas[index], positive_samples[index]
                for key in proposals.keys():
                    del proposals[key][index]
            sgs_start = sgs_start[index_0_reverse]

        proposals['padded_proposal_features'] = torch.nn.utils.rnn.pad_sequence(
            proposals['proposal_features'], batch_first=True)

        batch_size, device = len(sgs_start), sgs_start[0].device

        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')

        # noises = [torch.randn_like(sgs_start[i]) for i in range(len(sgs_start))]
        noises = torch.randn_like(sgs_start)
        # forward process ending
        sgs_t = self.q_sample(sgs_start, ts, noises)

        conditions = dict(feature_maps=feature_maps,
                          proposals=proposals)

        if self.diffusion_cfg.proposal_refine:
            # todo: need to refine this part to verify the validity of proposal refinement
            padded_proposal_features = conditions['proposals']['padded_proposal_features']
            refine_inst_cls = self.inst_cls_embed(padded_proposal_features)

        if len(index_0) > 0:
            aaa = 1.0
            pass

        pred_noises, attn_maps, pad_masks = self.pred_noise_model(sgs_t, ts, conditions, inst_num, img_metas)

        # pred_noises = [pred_noises[i][:inst_num[i], :inst_num[i], :] for i in range(batch_size)]

        # assert pred_noises.shape == noises.shape == sgs_start.shape
        # temp = torch.nn.functional.mse_loss(noises[0], pred_noises[0], reduction='mean')
        # temp = sgs_start[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :]

        #mse_loss = sum(torch.nn.functional.mse_loss(noises[i][:inst_num[i], :inst_num[i], :],
        #    pred_noises[i][:inst_num[i], :inst_num[i], :], reduction='mean') for i in range(batch_size)) / batch_size
        # mse_loss = [torch.nn.functional.mse_loss(sgs_start[i][:inst_num[i], :inst_num[i], :],
        #                 pred_noises[i][:inst_num[i], :inst_num[i], :], reduction='mean') for i in range(batch_size)]
        mse_loss = [torch.nn.functional.mse_loss(sgs_start[i][positive_samples[i][:, 0], positive_samples[i][:, 1], :],
                          pred_noises[i][positive_samples[i][:, 0], positive_samples[i][:, 1], :], reduction='mean')
                    for i in range(batch_size)]
        # mse_loss = sum(mse_loss) / batch_size
        mse_loss = torch.stack(mse_loss)
        # mse_loss = torch.nn.functional.mse_loss(noises, pred_noises, reduction='mean')
        '''for i in range(batch_size):
            if mse_loss[i] < 0.2:
                temp_loss_train = mse_loss[i]
                temp_sgs_start = sgs_start[i][positive_samples[i][:, 0], positive_samples[i][:, 1], :]
                temp_pred_noises = pred_noises[i][positive_samples[i][:, 0], positive_samples[i][:, 1], :]
                temp_loss = torch.nn.functional.mse_loss(temp_sgs_start, temp_pred_noises)
                temp_sgs_start = temp_sgs_start.cpu().numpy()
                temp_pred_noises = temp_pred_noises.cpu().detach().numpy()
                pass'''

        # check for debug
        '''for item in mse_loss:
            if math.isnan(item):
                print(inst_num)
                raise ValueError(f'loss value is NaN!!!')'''

        if self.diffusion_cfg.reweight:
            if self.diffusion_cfg.mean_type == 'x0':
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse_loss
            elif self.diffusion_cfg.mean_type == 'epsilon':
                weight = (1 - self.alphas_cumprod[ts]) / ((1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((sgs_start - self._predict_xstart_from_eps(sgs_t, ts, pred_noises)) ** 2 / 2.0)
                loss = torch.where((ts == 0), likelihood, mse_loss)
        else:
            loss = mse_loss
            weight = torch.tensor([1.0] * batch_size).to(device)

        terms = {}
        terms["loss"] = weight * loss

        terms["loss"] /= pt
        return terms

    def forward_test(self, imgs, img_metas, **kwargs):
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

        return self.diffusion_inference(imgs[0],
                                         img_metas[0],
                                         key_first=key_first,
                                         **kwargs)

    def diffusion_inference(self,
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
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes[0]
        if gt_labels is not None:
            gt_labels = gt_labels[0]
        if gt_masks is not None:
            gt_masks = gt_masks[0]

        x = self.extract_feat(img)

        device = x[0].device

        if gt_labels is not None:
            gt_labels = [label + 1 for label in gt_labels]

        (proposals, feature_maps, seg_results) = self.process_segment(x,
                                                                      img_meta,
                                                                      gt_bboxes,
                                                                      gt_labels,
                                                                      gt_masks,
                                                                      use_gt_box=self.diffusion_cfg.use_gt_box,                                                        use_gt_label=self.diffusion_cfg.use_gt_label,
                                                                      rescale=rescale)

        feature_maps = x if self.diffusion_cfg.multiscale_feature_map else feature_maps

        proposals['padded_proposal_features'] = torch.nn.utils.rnn.pad_sequence(
            proposals['proposal_features'], batch_first=True)

        inst_num = proposals['padded_proposal_features'].size(1)

        gt_list = None
        if gt_labels is not None:
            gt_list = {}
            gt_list['gt_labels'] = gt_labels
            gt_list['gt_masks'] = gt_masks
            gt_list['gt_rels'] = gt_rels[0]
        gt_sg, _, gt_clses, _, gt_sg_rels, positive_samples = self.get_gtscenegraph(gt_list['gt_labels'],
                                                                                    gt_list['gt_rels'],
                                                                                    gt_list['gt_masks'], proposals,
                                                                                    img_meta)
        ts, pt = self.sample_timesteps(1, device, 'uniform')
        ts[0] = 100
        noises = torch.randn_like(gt_sg)
        sgs_t = self.q_sample(gt_sg, ts, noises)

        noises = torch.randn((1, inst_num, inst_num, self.diffusion_cfg.prototype_dim)).to(device)

        conditions = dict(feature_maps=feature_maps,
                          proposals=proposals)

        # check for single step in testing
        t = torch.tensor([ts[0]] * sgs_t.shape[0]).to(sgs_t.device)
        temp_inst_num = [sgs_t.size(1)]
        out, attn_map, pad_mask = self.pred_noise_model(noises, t, conditions, temp_inst_num, img_meta)
        if self.diffusion_cfg.mean_type == 'epsilon':
            out = self._predict_xstart_from_eps(sgs_t, t, eps=out)
        mse_loss_1 = torch.nn.functional.mse_loss(gt_sg[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                  out[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                  reduction='mean')

        x_start = self.p_sample(sgs_t, conditions, self.diffusion_cfg.steps, img_meta, gt_sg, positive_samples)

        det_result = self.post_process(x_start, proposals, img_meta, gt_list)

        return det_result

    def get_gtscenegraph(self, gt_labels, gt_rels, gt_masks=None, proposals=None, img_metas=None):
        assert len(gt_labels) == len(gt_rels)
        batch_num, device = len(gt_labels), gt_labels[0].device
        if self.train_cfg is not None:
            mask_assigner = build_assigner(self.train_cfg.mask.assigner)
        else:
            mask_assigner = build_assigner(self.test_cfg.mask.assigner)
        proposal_clses, proposal_cls_preds, proposal_masks, proposal_mask_preds = proposals['proposal_clses'], \
            proposals['proposal_cls_preds'], proposals['proposal_masks'], proposals['proposal_mask_preds']
        sgs_start, gt_clses, sgs_rels, positive_samples = [], [], [], []

        for i, (gt_label, gt_rel, gt_mask, proposal_cls, proposal_cls_pred, proposal_mask, proposal_mask_pred,
                img_meta) in enumerate(
                zip(gt_labels, gt_rels, gt_masks, proposal_clses, proposal_cls_preds, proposal_masks,
                    proposal_mask_preds, img_metas)):
            gt_inst_num, gt_rel_num, pred_inst_num = gt_label.size(0), gt_rel.size(0), proposal_cls.size(0)
            # proposal_cls_onehot = F.one_hot(proposal_cls, num_classes=self.num_classes).to(device)
            gt_mask = torch.tensor(gt_mask.masks).to(device).to(torch.float32)
            if not self.training:
                ori_height, ori_width = img_meta['ori_shape'][:2]
                gt_mask = F.interpolate(
                    gt_mask[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

                gt_mask = gt_mask.to(torch.uint8)

            mask_assign_result = mask_assigner.assign(
                proposal_cls_pred[:, :-1], # for certain assignment, using 'proposal_cls_onehot.type(torch.float)' instead
                proposal_mask_pred, # for certain assignment, using 'proposal_mask.type(torch.float)' instead
                gt_label - 1,
                # torch.tensor(gt_mask.masks).to(device),
                gt_mask,
                img_meta,
            )
            assign_index = torch.nonzero(mask_assign_result.labels >= 0).view(-1)
            gt_index, gt_cls = mask_assign_result.gt_inds[assign_index] - 1, mask_assign_result.labels
            gt2assign = torch.full([gt_inst_num], -1).to(device).type(torch.long)
            if self.diffusion_cfg.proposal_assign_gt:
                sg_start = torch.zeros(
                    (pred_inst_num + 1, pred_inst_num + 1, self.diffusion_cfg.prototype_dim),
                    dtype=self.rel_prototype.dtype,
                    device=device,
                )
                positive_sample = torch.zeros(
                    (pred_inst_num + 1, pred_inst_num + 1),
                    dtype=torch.bool,
                    device=device,
                )
                gt2assign[gt_index] = assign_index
                sg_start[gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)], :] = \
                    self.rel_prototype.to(device)[gt_rel[:, 2].type(torch.long)]
                sg_start = sg_start[:-1, :-1, :]
                positive_sample[
                    gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)]] = True
                positive_sample = positive_sample[:-1, :-1]
                if not self.training:
                    sg_rels = torch.zeros(
                        (pred_inst_num + 1, pred_inst_num + 1),
                        dtype=self.rel_prototype.dtype,
                        device=device,
                    )
                    sg_rels[gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)]] = \
                        gt_rel[:, 2].type(torch.float32)
                    sg_rels = sg_rels[:-1, :-1]
            else:
                # in most case, assign_index.size(0) == gt_inst_num due to proposals >> GT,
                # but sometimes, assign_index.size(0) < gt_inst_num
                sg_start = torch.zeros(
                    (assign_index.size(0) + 1, assign_index.size(0) + 1, self.diffusion_cfg.prototype_dim),
                    dtype=self.rel_prototype.dtype,
                    device=device,
                )
                positive_sample = torch.zeros(
                    (assign_index.size(0) + 1, assign_index.size(0) + 1),
                    dtype=torch.bool,
                    device=device,
                )
                gt_cls = gt_cls[assign_index]
                continue_index = torch.arange(assign_index.size(0)).to(device)
                gt2assign[gt_index] = continue_index
                sg_start[gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)], :] = \
                    self.rel_prototype.to(device)[gt_rel[:, 2].type(torch.long)]
                sg_start = sg_start[:-1, :-1, :]
                positive_sample[
                    gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)]] = True
                positive_sample = positive_sample[:-1, :-1]
                if not self.training:
                    sg_rels = torch.zeros(
                        (assign_index.size(0) + 1, assign_index.size(0) + 1),
                        dtype=self.rel_prototype.dtype,
                        device=device,
                    )
                    sg_rels[gt2assign[gt_rel[:, 0].type(torch.long)], gt2assign[gt_rel[:, 1].type(torch.long)]] = \
                        gt_rel[:, 2].type(torch.float32)
                    sg_rels = sg_rels[:-1, :-1]
                proposals['proposal_clses'][i] = proposals['proposal_clses'][i][assign_index]
                proposals['proposal_cls_preds'][i] = proposals['proposal_cls_preds'][i][assign_index]
                proposals['proposal_masks'][i] = proposals['proposal_masks'][i][assign_index]
                proposals['proposal_mask_preds'][i] = proposals['proposal_mask_preds'][i][assign_index]
                proposals['proposal_features'][i] = proposals['proposal_features'][i][assign_index]

            sgs_start.append(sg_start)
            gt_clses.append(gt_cls)
            positive_samples.append(torch.nonzero(positive_sample))
            if not self.training:
                sgs_rels.append(sg_rels)

        inst_num = [sgs_start[i].size(0) for i in range(batch_num)]
        pad_len = max(inst_num)
        sgs_start = [
            F.pad(x, pad=(0, 0, 0, pad_len - x.shape[1], 0, pad_len - x.shape[0]), mode='constant',
                  value=0) for x in sgs_start]
        sgs_start = torch.stack(sgs_start)

        return sgs_start, proposals, gt_clses, inst_num, sgs_rels, positive_samples

    def p_sample(self, x_t, conditions, steps, img_meta, gt_sg, positive_samples):
        assert steps <= self.diffusion_cfg.steps, "The number of steps has exceeded the set limit."
        indices = list(range(250))[::-1]
        inst_num = [x_t.size(1)]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_t.device)
            out, attn_map, pad_mask = self.pred_noise_model(x_t, t, conditions, inst_num, img_meta)
            if self.diffusion_cfg.mean_type == 'epsilon':
                out = self._predict_xstart_from_eps(x_t, t, eps=out)
            mse_loss_1 = torch.nn.functional.mse_loss(gt_sg[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                    out[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                    reduction='mean')
            out_numpy = out.cpu().numpy()
            x_t = self.p_mean_variance(out, x_t, t)
            mse_loss_2 = torch.nn.functional.mse_loss(gt_sg[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                    x_t[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :],
                                                    reduction='mean')
            x_t_numpy = x_t[0].cpu().numpy()
        return x_t

    def p_mean_variance(self, pred, x_t, t):
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        return posterior_mean

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def post_process(self, x_start, proposal, img_meta, gt_list=None):
        def cosine_similarity(x1, x2, eps=1e-8):
            x1_norm = torch.norm(x1, p=2, dim=-1, keepdim=True)
            x2_norm = torch.norm(x2, p=2, dim=-1, keepdim=True)
            dot_product = torch.einsum('bijc,lc->bijl', x1, x2)
            union_norm = torch.matmul(x1_norm, x2_norm.T)
            # torch.sum(x1 * x2, dim=-1, keepdim=True)
            cos_sim = dot_product / (union_norm + eps)
            return cos_sim
        # similarity_score = torch.einsum('bijc,lc->bijl', x_start, self.rel_prototype.to(x_start.device))
        similarity_score = cosine_similarity(x_start, self.rel_prototype.to(x_start.device))
        similarity_score = torch.softmax(similarity_score, -1)
        temp = similarity_score.cpu().numpy()

        _, temp_1 = similarity_score.max(-1)
        temp_1 = temp_1.cpu().numpy()

        if gt_list is not None:
            gt_sg, _, gt_clses, _, gt_sg_rels, positive_samples = self.get_gtscenegraph(gt_list['gt_labels'], gt_list['gt_rels'], gt_list['gt_masks'], proposal, img_meta)
            pass

        gt_sg_positive = gt_sg[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :]
        x_start_positive = x_start[0][positive_samples[0][:, 0], positive_samples[0][:, 1], :]

        mse_loss = torch.nn.functional.mse_loss(gt_sg_positive, x_start_positive, reduction='mean')

        self.visual_result(temp_1[0], proposal['proposal_clses'][0])

        if self.diffusion_cfg.union_inference:
            return self.union_proposal_inference(x_start, similarity_score, proposal, img_meta)
        else:
            return self.certain_proposal_inference(x_start, similarity_score, proposal, img_meta)

    def certain_proposal_inference(self, x_start, similarity_score, proposal, img_meta):

        det_result = Result(
            seg_scores=None,
            pan_results=None,
            masks=None,
            refine_bboxes=None,
            refine_dists=None,
            refine_scores=None,
            refine_labels=None,
            rel_dists=None,
            rel_pair_idxes=None,
            rels=None,
        )

        return det_result

    def union_proposal_inference(self, x_start, similarity_score, proposal, img_meta):
        # todo: achieve scene graph prediction with relation-constrained
        det_result = Result(
            seg_scores=None,
            pan_results=None,
            masks=None,
            refine_bboxes=None,
            refine_dists=None,
            refine_scores=None,
            refine_labels=None,
            rel_dists=None,
            rel_pair_idxes=None,
            rels=None,
        )

        return det_result

    def visual_result(self, pred_map, inst_cls):
        inst_num = pred_map.shape[0]
        for i in range(inst_num):
            for j in range(inst_num):
                if pred_map[i][j] != 0 and pred_map[i][j] != 55:
                    print(self.inst_classes[inst_cls[i]], " ", self.rel_classes[pred_map[i][j]], " ", self.inst_classes[inst_cls[j]])


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                  produces the cumulative product of (1-beta) up to that
                  part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                 prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)