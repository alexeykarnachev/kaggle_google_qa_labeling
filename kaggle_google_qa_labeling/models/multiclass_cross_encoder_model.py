import torch
import torch.nn as nn
from kaggle_google_qa_labeling.models.utilities import get_h, get_hid_size


class MulticlassCrossEncoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, mask_val, y_dim, loss_fn, pooling, clf_hid_dim, dpt):
        super().__init__()
        self.mask_val = mask_val
        self.encoder = encoder
        self.hid_size = get_hid_size(self.encoder, pooling)
        self.y_dim = y_dim

        self.clfs = []
        for i in range(y_dim[0]):
            if clf_hid_dim is None:
                clf = nn.Sequential(
                    nn.Dropout(dpt),
                    nn.Linear(self.hid_size, self.y_dim[1])
                )
            else:
                clf = nn.Sequential(
                    nn.Dropout(dpt),
                    nn.Linear(self.hid_size, clf_hid_dim),
                    nn.ReLU(),
                    nn.Dropout(dpt),
                    nn.Linear(clf_hid_dim, self.y_dim[1])
                )

            self.clfs.append(clf)

        self.clfs = nn.ModuleList(self.clfs)

        self.loss_fn = loss_fn

        self.pooling = pooling.lower()

    def get_head_parameters(self):
        return list(self.clfs.parameters())

    def forward(self, data, targets):
        unpacked_data, token_types = data
        y = targets[0] if targets is not None else None

        result = [0, []]

        for i in range(len(unpacked_data)):
            h = get_h(self.encoder, unpacked_data[i], token_types[i], mask_val=self.mask_val, pooling=self.pooling)

            result_ = []

            for j in range(self.y_dim[0]):
                clf = self.clfs[j]
                logits_ = clf(h)

                if y is not None:
                    loss = self.loss_fn(logits_, y[:, j].long()) / len(unpacked_data) / self.y_dim[0]
                    result[0] += loss

                result_.append(logits_)

            result[1].append(torch.stack(result_, dim=-1))

        return result
