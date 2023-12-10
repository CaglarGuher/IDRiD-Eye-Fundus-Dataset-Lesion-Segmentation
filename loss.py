import torch.nn as nn

from segmentation_models_pytorch.base import base
import segmentation_models_pytorch.losses._functional as F
from segmentation_models_pytorch.base.modules import Activation

class WeightedCombinationLoss(base.Loss):
    def __init__(self, dice_weight=0.0, ce_weight=1.0, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

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

        # Weighted Combination Loss
        weighted_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return weighted_loss