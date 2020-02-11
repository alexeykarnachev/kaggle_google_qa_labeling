import torch
import torch.nn as nn

from kaggle_google_qa_labeling.models.utilities import get_h, get_hid_size


class BiEncoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dims, emb_feature_dims, mask_val, y_dim, loss_fn, pooling,
                 clf_hid_dim, dropout_rate):

        super().__init__()
        self.mask_val = mask_val
        self.encoder = encoder
        self.hid_size = get_hid_size(self.encoder, pooling) * 2
        self.y_dim = y_dim or 1

        if (feature_dims is not None) & (emb_feature_dims is not None):
            self.cat_embeddings = [
                nn.Embedding(
                    x,
                    y
                ) for x, y in zip(feature_dims, emb_feature_dims)
            ]
            self.cat_embeddings = nn.ModuleList(self.cat_embeddings)
            self.emb_feature_dims = emb_feature_dims
        else:
            self.cat_embeddings = None
            self.emb_feature_dims = [0]

        if clf_hid_dim is None:
            self.clf = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.hid_size + sum(self.emb_feature_dims), self.y_dim)
            )
        else:
            self.clf = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.hid_size + sum(self.emb_feature_dims), clf_hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(clf_hid_dim, self.y_dim)
            )

        self.loss_fn = loss_fn

        self.pooling = pooling.lower()

    def get_head_parameters(self):
        return list(self.clf.parameters())

    def forward(self, data, targets):
        # X - list of four lists
        # TODO: ...
        # qX - question X
        # aX - answer X
        # qT - question token type
        # aT - answer token type

        if self.cat_embeddings is None:
            qX, aX, qT, aT = data
        else:
            qX, aX, qT, aT, F = data

        # y is list of some variants of crops
        y = targets[0] if targets is not None else None

        result = [0, []]

        for i in range(len(qX)):
            qh = get_h(self.encoder, qX[i], qT[i], mask_val=self.mask_val, pooling=self.pooling)
            ah = get_h(self.encoder, aX[i], aT[i], mask_val=self.mask_val, pooling=self.pooling)
            h = torch.cat([qh, ah], dim=-1)

            if self.cat_embeddings is not None:
                for e in range(len(self.emb_feature_dims)):
                    feature_emb = self.cat_embeddings[e](F[:, e])
                    h = torch.cat([h, feature_emb], dim=1)

            logits = self.clf(h)

            if y is not None:
                loss = self.loss_fn(logits, y) / len(qX)
                result[0] += loss

            result[1].append(logits)

        return result
