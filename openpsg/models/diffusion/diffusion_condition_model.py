import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.core import BitmapMasks, bbox2roi, build_assigner, multiclass_nms
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import HEADS, builder
from mmdet.models.builder import build_head
from mmcv.cnn import normal_init, xavier_init
import math

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap

from ..detectors.panseg import PanSeg
from ..detectors.detr_plus import DETR_plus

INF = 1e8


@HEADS.register_module()
class DiffusionConditionModel(nn.Module):
    def __init__(self, config):
        super(DiffusionConditionModel, self).__init__()
        self.config = config

        self.step_embed_layer = nn.Linear(self.config.step_embed_dim, self.config.step_embed_dim)
        '''self.step_embed_layer = nn.Sequential(
            nn.linear(self.config.step_embed_dim, self.config.step_embed_dim * 2),
            nn.SiLU(),
            nn.linear(self.config.step_embed_dim * 2, self.config.step_embed_dim * 2),
        )'''

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.config.step_embed_dim + self.config.prototype_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        self.compress_state = nn.Linear(self.config.hidden_dim, self.config.prototype_dim)

        self.condition_cross_attn = ConditionCrossAttn(self.config)

        self.condition_cross_attn = nn.ModuleList([
            ConditionCrossAttn(self.config) for _ in range(self.config.layer_num)
        ])

        self.init_weights()

    def init_weights(self):
        xavier_init(self.step_embed_layer)
        xavier_init(self.compress_state)

    def forward(self, sgs_t, time_steps, conditions, inst_num, img_metas):
        batch_size, device = time_steps.size(0), time_steps.device
        time_embed = timestep_embedding(time_steps, self.config.step_embed_dim).to(device)
        time_embed = self.step_embed_layer(time_embed).unsqueeze(1).unsqueeze(1)

        pad_len = max(inst_num)
        '''sgs_t = [
            F.pad(x, pad=(0, 0, 0, pad_len - x.shape[1], 0, pad_len - x.shape[0]), mode='constant',
                  value=0) for x in sgs_t]
        sgs_t = torch.stack(sgs_t)'''

        fusion_state = torch.cat([sgs_t, time_embed.expand(-1, pad_len, pad_len, -1)], dim=-1)

        # fusion_state = [torch.cat([sgs_t[i], time_embed[i].expand(sgs_t[i].size(0), sgs_t[i].size(1), -1)], dim=-1) for i in range(batch_size)]

        fusion_state = self.fusion_layer(fusion_state)

        inst_num_expand = torch.LongTensor(inst_num).to(device).unsqueeze(1).expand(-1, pad_len)
        attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(batch_size, -1).ge(
            inst_num_expand).unsqueeze(1).unsqueeze(1).expand(-1, pad_len, pad_len, -1)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(batch_size, -1).lt(
            inst_num_expand).unsqueeze(-1)
        non_pad_mask = (non_pad_mask & non_pad_mask.view(batch_size, 1, pad_len))

        for condition_layer in self.condition_cross_attn:
            fusion_state, attn_map, pad_mask = condition_layer(fusion_state, conditions, attn_mask, non_pad_mask, img_metas)

        fusion_state = self.compress_state(fusion_state)

        return fusion_state, attn_map, pad_mask


@HEADS.register_module()
class ConditionCrossAttn(nn.Module):
    def __init__(self, config):
        super(ConditionCrossAttn, self).__init__()
        self.config = config
        self.global_condition = self.config.global_condition
        self.input_dim = self.config.input_dim
        self.hidden_dim = self.config.hidden_dim
        self.prototype_dim = self.config.prototype_dim
        self.num_heads = self.config.num_heads

        self.w_qs = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_ks = nn.Linear(self.input_dim, self.hidden_dim)
        self.w_vs = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.hidden_dim + self.hidden_dim // self.num_heads)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.input_dim + self.hidden_dim // self.num_heads)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.input_dim + self.hidden_dim // self.num_heads)))

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc.weight)

        self.w1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, 1))
        self.w2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, 1))

        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, conditions, attn_mask, non_pad_mask, img_metas):
        proposals = conditions['proposals']
        proposal_features = proposals['padded_proposal_features']
        assert proposal_features.size(2) == self.input_dim
        if self.global_condition:
            # todo: add implement for global feature maps as extra conditions
            feature_maps = conditions['feature_maps']

        batch_size, inst_num, _, _ = x.size()
        single_dim = self.hidden_dim // self.num_heads

        q = self.w_qs(x).view(batch_size, inst_num, inst_num, self.num_heads, single_dim)
        k = self.w_ks(proposal_features).view(batch_size, inst_num, self.num_heads, single_dim)
        v = self.w_vs(proposal_features).view(batch_size, inst_num, self.num_heads, single_dim)

        q = q.permute(3, 0, 1, 2, 4).contiguous().view(-1, inst_num, inst_num, single_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, inst_num, single_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, inst_num, single_dim)

        attn_mask = attn_mask.repeat(self.num_heads, 1, 1, 1)
        non_pad_mask = non_pad_mask.unsqueeze(-1)

        # q: (8 * batch_size, inst_num, inst_num, 64)
        # k: (8 * batch_size, inst_num, 64)
        # v: (8 * batch_size, inst_num, 64)
        attn = torch.einsum('bijd,bkd->bijk', q, k) / np.power(single_dim, 0.5)
        attn = attn.masked_fill(attn_mask, -INF) # -np.inf

        attn = self.softmax(attn)
        attn = self.dropout1(attn)

        output = torch.einsum('bijk,bkd->bijd', attn, v)

        output = output.view(self.num_heads, batch_size, inst_num, inst_num, single_dim)
        output = output.permute(1, 2, 3, 0, 4).contiguous().view(batch_size, inst_num, inst_num, -1)

        output = self.dropout2(self.fc(output))
        output = self.layer_norm1(output + x)

        output *= non_pad_mask.float()

        residual = output
        output = output.permute(0, 3, 1, 2).contiguous()
        output = self.w2(F.relu(self.w1(output)))
        output = output.permute(0, 2, 3, 1).contiguous()
        output = self.layer_norm2(self.dropout3(output) + residual)
        output *= non_pad_mask.float()

        return output, attn, non_pad_mask


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding