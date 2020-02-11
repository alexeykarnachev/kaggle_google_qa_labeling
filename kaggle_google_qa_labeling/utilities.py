import os
import json
import gzip
import pickle
import random
import hashlib
import datetime

from html import unescape
from logging.handlers import RotatingFileHandler

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from kaggle_google_qa_labeling.custom_json_encoder import CustomJSONEncoder


def load_json(file):
    with open(file) as f:
        return json.load(f)


def load_object(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dump_object(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


class CustomRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        super(CustomRotatingFileHandler, self).doRollover()
        old_log = self.baseFilename + ".1"
        with open(old_log, 'rb') as log:
            now = datetime.now().strftime("%d-%m-%y-%H:%M:%S")
            with gzip.open(self.baseFilename + now + '.gz', 'wb') as comp_log:
                comp_log.writelines(log)
        os.remove(old_log)


def dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=CustomJSONEncoder, indent=2, ensure_ascii=False)


def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(l):
    return [item for sublist in l for item in sublist]


def pad_sequences(seqs, max_len, padding, pad_val):
    if not seqs:
        return np.array([])

    if padding not in ['post', 'pre']:
        raise ValueError("Padding must be 'post' or 'pre'")

    max_len = max([len(x) for x in seqs]) if not max_len else max_len

    new_seqs = []

    for seq in seqs:

        if padding == 'post':
            new_seq = list(seq[-max_len:])
            new_seq = new_seq + [pad_val] * (max_len - len(new_seq))
        else:
            new_seq = list(seq[:max_len])
            new_seq = [pad_val] * (max_len - len(new_seq)) + new_seq

        new_seqs.append(new_seq)

    return np.vstack(new_seqs)


def get_cur_time_str():
    return datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S__%f')[:-3]


def tokenize_text(t, tokenizer: PreTrainedTokenizer):
    t = unescape(t)
    t = t.replace('<br>', '\n')
    t = tokenizer.encode(t, add_special_tokens=False)
    return t


def get_bert_parameters(model, freeze_emb: bool):
    param_optimizer = np.array(list(model.named_parameters()))
    if freeze_emb:
        zero_grad_mask = np.array(['embeddings' in x[0] for x in param_optimizer])
        param_optimizer = param_optimizer[~zero_grad_mask]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters
