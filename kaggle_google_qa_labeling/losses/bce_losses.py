import torch
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss


class BCESoftLoss(_Loss):
    __reductions__ = {'mean', 'sum'}

    def __init__(self, reduction='mean', y_dim_reduction='mean', ignore_index=-1):
        super(BCESoftLoss, self).__init__(reduction=reduction)
        self.reduction = reduction
        self.y_dim_reduction = y_dim_reduction
        self.e = 1e-6
        self.ignore_index = ignore_index

        if self.reduction not in BCESoftLoss.__reductions__ or self.y_dim_reduction not in BCESoftLoss.__reductions__:
            raise ValueError(f'Only {BCESoftLoss.__reductions__} reductions are supported')

    def forward(self, x, y):
        x = torch.clamp(x, self.e, 1 - self.e)
        y = torch.clamp(y, self.e, 1 - self.e)
        mask = y != self.ignore_index

        loss = y * torch.log(y / x) + (1 - y) * torch.log((1 - y) / (1 - x))
        loss[~mask] = 0
        loss = loss.squeeze(-1)

        if self.reduction == 'mean':
            loss_reduced = loss.sum(dim=0) / (mask.sum(dim=0) + self.e)
        elif self.reduction == 'sum':
            loss_reduced = loss.sum(dim=0)

        if len(loss_reduced.size()) > 0:
            if self.y_dim_reduction == 'mean':
                loss_reduced = loss_reduced.mean()
            elif self.y_dim_reduction == 'sum':
                loss_reduced = loss_reduced.sum()

        return loss_reduced


class BCESoftLossFromLogits(BCESoftLoss):
    def __init__(self, reduction='mean', y_dim_reduction='mean', ignore_index=-1):
        super().__init__(reduction=reduction, y_dim_reduction=y_dim_reduction, ignore_index=ignore_index)

    def forward(self, x, y):
        x = torch.sigmoid(x)
        return super().forward(x, y)


BCELossFromLogits = BCEWithLogitsLoss
