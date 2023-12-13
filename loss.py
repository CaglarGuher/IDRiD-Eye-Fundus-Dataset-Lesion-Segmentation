import torch.nn as nn
from segmentation_models_pytorch.utils import base
import segmentation_models_pytorch.utils.functional as F
from segmentation_models_pytorch.base.modules import Activation
import torch
class FocalLoss(_WeightedLoss):
    def __init__(self, alpha=None, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_factor=None):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.balance_factor = balance_factor

    def forward(self, input, target):
        # Compute cross_entropy (neg log likelihood)
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction,
                                  ignore_index=self.ignore_index)

        # Compute focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class balance (if balance_factor is specified)
        if self.balance_factor is not None:
            balanced_focal_loss = self.balance_factor * focal_loss
            return balanced_focal_loss

        return focal_loss

class WeightedCombinationLoss(nn.Module):
    def __init__(self, dice_weight=0.0, ce_weight=1.0, focal_weight=0.0, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super(WeightedCombinationLoss, self).__init__(**kwargs)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.eps = eps
        self.beta = beta
        self.activation = nn.Sigmoid() if activation is None else activation
        self.ignore_channels = ignore_channels

        self.focal_loss = FocalLoss()

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        # Dice Loss
        dice_loss = 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        y_pr = y_pr.squeeze(1)

        # Cross-Entropy Loss
        ce_loss = nn.BCELoss()(y_pr, y_gt)

        # Focal Loss
        focal_loss = self.focal_loss(y_pr, y_gt)

        # Weighted Combination Loss
        weighted_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss + self.focal_weight * focal_loss

        return weighted_loss