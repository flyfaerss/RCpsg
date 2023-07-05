import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


#@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def dice_loss(input, target, mask=None, eps=0.001):
    N, H, W = input.shape

    input = input.contiguous().view(N, H * W)
    target = target.contiguous().view(N, H * W).float()
    if mask is not None:
        mask = mask.contiguous().view(N, H * W).float()
        input = input * mask
        target = target * mask
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    #print('1-d max',(1-d).max())
    return 1 - d


@LOSSES.register_module()
class psgtrDiceLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(psgtrDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0

    def forward(self, inputs, targets, num_matches):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return self.loss_weight * loss.sum() / num_matches


@LOSSES.register_module()
class MultilabelCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss = -(F.log_softmax(inputs, dim=1) *
                 targets).sum(1) / targets.sum(1)
        loss = loss.mean()
        return self.loss_weight * loss


@LOSSES.register_module()
class RelLabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, use_peseudo_labels=False):
        super(RelLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = torch.nn.KLDivLoss()
        self.use_peseudo_labels = use_peseudo_labels

    def add_soft_labels(self, pred, target, hard_index, resistance_bias, fusion_weight):
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # new_pred = pred # - resistance_bias # * fusion_weight

        pred_label = torch.argmax(pred, dim=1)
        count = 0
        for i in range(pred_label.shape[0]):
            if pred_label[i] in hard_index and target[i] == 0:
                rest_score = true_dist[i, 0] + true_dist[i, pred_label[i]]
                true_dist[i, 0] -=0.1 # = rest_score / 2.0 # -= 0.2
                true_dist[i, pred_label[i]] += 0.1 # rest_score / 2.0 # += 0.2
                count += 1
        if count != 0:
            print(count)
        return true_dist

    def forward(self, pred, target, hard_index=None, resistance_bias=None, fusion_weight=None):
        pred = pred.log_softmax(dim=self.dim)

        if self.use_peseudo_labels:
            assert hard_index is not None and resistance_bias is not None
            true_dist = self.add_soft_labels(pred, target, hard_index, resistance_bias, fusion_weight)
        else:
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        # loss = self.loss(pred, true_dist)

        return loss


@LOSSES.register_module()
class DynamicReweightCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weight):
        if len(torch.nonzero(targets)) == 0:
            loss_relation = None
        else:
            criterion_loss = nn.CrossEntropyLoss(weight=weight)
            loss_relation = criterion_loss(inputs[targets != -1],
                                               targets[targets != -1].long())
            return loss_relation


@LOSSES.register_module()
class MultilabelLogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        # loss = loss.mean()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())


@LOSSES.register_module()
class LogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        positive_rate = 50
        loss_1 = -(torch.log(
            (inputs + 1) / 2 + 1e-14) * targets).sum() * positive_rate
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())

    # def forward(self, inputs, targets):
    #     loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
    #     return self.loss_weight * loss_1

    # def forward(self, inputs, targets):
    #     inputs  = (inputs + 1) / 2 + 1e-14
    #     loss = F.mse_loss(inputs, targets.float(), reduction='mean')
    #     return self.loss_weight * loss


@LOSSES.register_module()
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum', loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, num_matches):

        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                     targets,
                                                     reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return self.loss_weight * loss.mean(1).sum() / num_matches

        # pt = torch.sigmoid(_input)
        # bs = len(pt)
        # target = target.type(torch.long)
        # # print(pt.shape, target.shape)
        # alpha = self.alpha
        # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        #     (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # # print('loss_shape',loss.shape)
        # if self.reduction == 'elementwise_mean':
        #   loss = torch.mean(loss)
        # elif self.reduction == 'sum':
        #   loss = torch.sum(loss)

        # return loss*self.loss_weight/bs


def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor


def l2_norm(feature, axis=1):
    norm = torch.norm(feature,2,axis,True)
    output = torch.div(feature, norm)
    return output


def loss_eval_feature(feature, labels, embed, weight, tau=0.1):
    feature = l2_norm(feature, axis=1)
    embed = l2_norm(embed, axis=1)
    fg_labels = squeeze_tensor(torch.nonzero(labels[labels != -1]))
    valid_feature = feature[fg_labels]
    valid_labels = labels[fg_labels].long()
    if weight is not None:
        valid_weight = weight[valid_labels]

    labels_list = []
    for i in range(valid_labels.shape[0]):
        if valid_labels[i] not in labels_list:
            labels_list.append(valid_labels[i])
    labels_list = torch.tensor(labels_list).squeeze().long().to(feature.device)

    match_inner_product = torch.exp(torch.mul(valid_feature, embed[valid_labels]).sum(-1) / tau)
    all_inner_product = torch.exp(torch.mul(valid_feature.unsqueeze(1), embed[labels_list]).sum(-1) / tau).sum(-1)
    # loss = torch.mean(-torch.log(match_inner_product / (all_inner_product - match_inner_product)))
    if weight is not None:
        loss = torch.mean(-torch.log(match_inner_product / all_inner_product) * valid_weight)
    else:
        loss = torch.mean(-torch.log(match_inner_product / all_inner_product))

    return loss

@LOSSES.register_module()
class FeatureLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.tau = 0.1

    def forward(self, feature, labels, embed, weight):
        return loss_eval_feature(feature, labels, embed, weight, self.tau)


