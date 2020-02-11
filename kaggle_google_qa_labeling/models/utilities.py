import json
import re
from typing import Dict

import torch


def resize_token_type_embeddings(model, type_vocab_size):
    single_emb = model.embeddings.token_type_embeddings
    model.embeddings.token_type_embeddings = torch.nn.Embedding(type_vocab_size, single_emb.embedding_dim)
    model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([type_vocab_size, 1]))


def get_params(groups_lr: Dict[str, float], encoder, model):
    default_lr = groups_lr.get('default', 0) or 0
    params = []
    summary = {}

    for name, param in encoder.named_parameters():
        name = name.lower()

        if name.startswith('embeddings'):
            group_name = 'embeddings'
        elif re.match(r'^encoder\.layer\.\d+\..+', name):
            id_ = int(re.findall(r'\d+', name)[0])
            group_name = f"encoder_{id_}"
        elif name.startswith('pooler'):
            param.requires_grad = False
            summary[name] = 0
            continue
        else:
            group_name = name

        group_lr = groups_lr.get(group_name, default_lr)

        if group_lr > 0:
            param.requires_grad = True
            params.append({'params': param, 'lr': group_lr})
        else:
            param.requires_grad = False

        summary[name] = group_lr

    head_params = model.get_head_parameters()
    group_lr = groups_lr.get('head', default_lr)

    for param in head_params:
        if group_lr > 0:
            param.requires_grad = True
            params.append({'params': param, 'lr': group_lr})
        else:
            param.requires_grad = False

    summary['head'] = group_lr
    summary = json.dumps(summary, indent=2)
    print(f'Learning rates were assigned:\n{summary}\n')
    return params


def get_h(model, data, token_types, mask_val, pooling):
    try:
        res = model(data, attention_mask=(data != mask_val).int(), token_type_ids=token_types)
    except TypeError:
        res = model(data, attention_mask=(data != mask_val).int())

    n = re.findall(r'\d+', pooling)
    if re.match(r'average_concat:\d+', pooling):
        h = torch.cat([r.mean(1) for r in res[2][-int(n[0]):]], 1)
    elif re.match(r'average_average:\d+', pooling):
        h = torch.stack([r.mean(1) for r in res[2][-int(n[0]):]], 1).mean(1)
    elif re.match(r'max_max:\d+', pooling):
        h = torch.stack([r.max(1)[0] for r in res[2][-int(n[0]):]], 1).max(1)[0]
    elif re.match(r'max_concat:\d+', pooling):
        h = torch.cat([r.max(1)[0] for r in res[2][-int(n[0]):]], 1)
    elif re.match(r'max_average:\d+', pooling):
        h = torch.stack([r.max(1)[0] for r in res[2][-int(n[0]):]], 1).mean(1)
    elif pooling == 'cls':
        h = res[0][:, 0, :]
    elif pooling == 'average':
        h = res[0].mean(1)
    else:
        raise ValueError(f'Unknown pooling strategy: {pooling}')

    return h


def get_hid_size(model, pooling):
    if re.match(r'.*concat:\d+', pooling):
        n = int(re.findall(r'\d+', pooling)[0])
    else:
        n = 1

    hid_size = model.config.hidden_size * n

    return hid_size
