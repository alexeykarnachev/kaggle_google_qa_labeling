import torch
import torch.nn as nn

from kaggle_google_qa_labeling.models.utilities import get_h, get_hid_size


class BiEncoderQACoupledModel(nn.Module):
    def __init__(self, encoder: nn.Module, mask_val, y_dim, loss_fn, pooling, clf_hid_dim, dropout_rate):
        super().__init__()
        self.mask_val = mask_val
        self.encoder = encoder
        self.hid_size = get_hid_size(self.encoder, pooling)
        self.y_dim = y_dim or 1

        self.loss_fn = loss_fn

        self.pooling = pooling.lower()

        self.w = nn.Parameter(torch.randn(2, y_dim), requires_grad=True)
        self.clf_hid_dim = clf_hid_dim

        if self.clf_hid_dim is None:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(self.hid_size, 1)
                    )
                    for _ in range(y_dim)
                ]
            )
        else:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(self.hid_size, self.clf_hid_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(self.clf_hid_dim, 1)
                    )
                    for _ in range(y_dim)
                ]
            )

    def get_head_parameters(self):
        return list(self.classifiers.parameters()) + [self.w]

    def forward(self, data, targets):
        # X - list of four lists
        # TODO: ...
        qX, aX, qT, aT = data

        # y is list of some variants of crops
        y = targets[0] if targets is not None else None
        # TODO: ...
        result = [0, []]

        for i in range(len(qX)):
            qh = get_h(self.encoder, qX[i], qT[i], mask_val=self.mask_val, pooling=self.pooling)
            ah = get_h(self.encoder, aX[i], aT[i], mask_val=self.mask_val, pooling=self.pooling)

            h = torch.stack([qh, ah], dim=-1)
            ws = torch.softmax(self.w, 0)
            h = torch.unbind(h @ ws, dim=-1)

            logits = []
            for j in range(len(self.classifiers)):
                logits.append(self.classifiers[i](h[j]))

            logits = torch.cat(logits, dim=1)

            if y is not None:
                loss = self.loss_fn(logits, y) / len(qX)
                result[0] += loss

            result[1].append(logits)

        return result
