
import torch
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RelDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(RelDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                s_reg_branches=None,
                o_reg_branches=None,
                r_reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        # intermediate_s_reference_points = []
        # intermediate_o_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if r_reg_branches is not None:
                r_tmp = r_reg_branches[lid](output)
                # s_tmp = s_reg_branches[lid](output)
                # o_tmp = o_reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    '''s_new_reference_points = s_tmp + r_tmp + inverse_sigmoid(
                        reference_points)
                    o_new_reference_points = o_tmp + r_tmp + inverse_sigmoid(
                        reference_points)'''
                    new_reference_points= r_tmp + inverse_sigmoid(reference_points)
                    # s_new_reference_points = s_new_reference_points.sigmoid()
                    # o_new_reference_points = o_new_reference_points.sigmoid()
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    # s_new_reference_points = s_tmp + r_tmp
                    # o_new_reference_points = o_tmp + r_tmp
                    new_reference_points = r_tmp
                    '''s_new_reference_points[..., :2] = s_tmp[
                        ..., :2] + r_tmp[..., :2] + inverse_sigmoid(reference_points)
                    o_new_reference_points[..., :2] = o_tmp[
                        ..., :2] + r_tmp[..., :2] + inverse_sigmoid(reference_points)'''
                    new_reference_points[..., :2] = r_tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    # s_new_reference_points = s_new_reference_points.sigmoid()
                   #  o_new_reference_points = o_new_reference_points.sigmoid()
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)
            # intermediate_s_reference_points.append(s_new_reference_points)
            # intermediate_o_reference_points.append(o_new_reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points), None, None
            #torch.stack(intermediate_s_reference_points), torch.stack(intermediate_o_reference_points)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)