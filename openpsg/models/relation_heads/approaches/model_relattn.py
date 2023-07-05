
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor


class RelAttnContext(nn.Module):
    def __init__(self, cfg,):

        super(RelAttnContext, self).__init__()
        self.cfg = cfg

        self.obj_downdim_fc = nn.Sequential(
            nn.ReLU(),
            make_fc(self.pooling_dim, self.hidden_dim),
        )

        self.rel_downdim_fc = nn.Sequential(
            nn.ReLU(),
            make_fc(self.pooling_dim, self.hidden_dim),
        )

        self.proj_inst_fusion = nn.Sequential(
            make_fc(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            make_fc(self.hidden_dim, self.hidden_dim),
        )

        self.proj_rel_fusion = nn.Sequential(
            make_fc(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            make_fc(self.hidden_dim, self.hidden_dim),
        )

    def init_weights(self):

        pass

    def forward(
        self,
        mlvl_feats,
        img_meta,
        inst_feats,
        rel_feats,
        det_result,
    ):

        batch_size = mlvl_feats[0].size(0)

        input_img_h, input_img_w = img_meta[0]['batch_input_shape']

        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_meta[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_positional_encodings)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2) # flatten the feature map
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes,
                                         dtype=torch.long,
                                         device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(query=feat_flatten,
                              key=None,
                              value=None,
                              query_pos=lvl_pos_embed_flatten,
                              query_key_padding_mask=mask_flatten,
                              spatial_shapes=spatial_shapes,
                              reference_points=reference_points,
                              level_start_index=level_start_index,
                              valid_ratios=valid_ratios,
                              **kwargs)


        # get the raw feature from the detection model and fusion operation
        inst_raw_feature = self.obj_downdim_fc(augment_inst_feat)
        rel_raw_feature = self.rel_downdim_fc(rel_feats)

        rel_feats_iters = [rel_raw_feature]
        inst_feats_iters = [inst_raw_feature]
        pred_relatedness_score_iters = []

        inst_feats_final = inst_feats_iters[-1]
        rel_feats_final = rel_feats_iters[-1]


        return (inst_feats_final, rel_feats_final, self.relation_embed.to(inst_feats_final.device),
                pred_relatedness_score_iters[-1])
