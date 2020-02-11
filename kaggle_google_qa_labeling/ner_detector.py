import re
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.special import softmax
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset

from kaggle_google_qa_labeling.factory import Factory
from kaggle_google_qa_labeling.length_sort_sampler import LengthSortSampler
from kaggle_google_qa_labeling.utilities import get_chunks, pad_sequences, load_json
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import pad_sequences


class TokenClassificationDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, ind):
        if self.targets is not None:
            return self.data[ind], self.targets[ind]
        else:
            return (self.data[ind],)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_collate_fn(max_len, pad_id):
        def collate_fn(data):
            d = list(zip(*data))
            data = d[0]
            targets = d[1] if len(d) == 2 else None

            seq_len = min(max_len, max([len(x) for x in data]))

            data = torch.LongTensor(pad_sequences(data, seq_len, 'post', pad_id))

            if targets is not None:
                targets = torch.LongTensor(pad_sequences(targets, seq_len, 'post', pad_id))
                res = (data, targets)
            else:
                res = (data,)

            return res

        return collate_fn

    def get_data_loader(self, bs, max_len, pad_id, drop_last, use_length_sampler=True):

        if use_length_sampler:
            sampler_ = LengthSortSampler(self.data, bs=bs)
        else:
            sampler_ = None

        dl = DataLoader(self, batch_size=bs, collate_fn=self.get_collate_fn(max_len, pad_id),
                        sampler=sampler_, drop_last=drop_last)
        return dl


class NERDetector:
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer, max_len: int,
                 device: str, token: str, bs: int, threshold: float, min_span_len: Optional[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.token = token
        self.bs = bs
        self.threshold = threshold
        self.min_span_len = min_span_len

    @classmethod
    def from_model_dir(cls, model_dir: Path, device: str, bs: int, threshold: float,
                       min_span_len: Optional[int]) -> 'NERDetector':
        config = load_json(model_dir / 'description.json')
        tokenizer_cls = Factory.get_class(f"transformers.{config['tokenizer_cls']}")
        tokenizer = tokenizer_cls.from_pretrained(str(model_dir))
        max_len = config['max_len']
        token = config['token']
        model_file_path = str(Path(model_dir) / 'model.pth')
        model = torch.load(model_file_path)

        obj = cls(model=model, tokenizer=tokenizer, max_len=max_len, device=device, token=token, bs=bs,
                  threshold=threshold, min_span_len=min_span_len)

        return obj

    def __call__(self, texts: List[str]) -> List[str]:
        self.model.eval()

        slices = []
        data = []
        pb = tqdm(enumerate(texts), total=len(texts), desc='[0/3 NerDetector] Tokenize texts')
        for i, text in pb:
            encoded_sequence = self.tokenizer.encode(text, add_special_tokens=False)

            if len(encoded_sequence) == 0:
                encoded_sequence = [self.tokenizer.pad_token_id]

            encoded_sequence = [self.tokenizer.build_inputs_with_special_tokens(x) for x in
                                get_chunks(encoded_sequence, self.max_len - 2)]

            if len(slices) == 0:
                slices.append(slice(0, len(encoded_sequence)))
            else:
                slices.append(slice(slices[-1].stop, slices[-1].stop + len(encoded_sequence)))

            data.extend(encoded_sequence)

        ds = TokenClassificationDataset(data, None)
        dl = ds.get_data_loader(self.bs, self.max_len, self.tokenizer.pad_token_id,
                                drop_last=False, use_length_sampler=True)

        targets = []
        data = []
        pb = tqdm(dl, total=len(dl), desc='[1/3 NerDetector] Evaluate model')
        backsort_inds = np.argsort(dl.sampler.inds)
        with torch.no_grad():
            for encoded_sequence in pb:
                encoded_sequence = encoded_sequence[0].to(self.device)
                slided_targets = self.model(encoded_sequence)[0].detach().cpu().numpy()
                encoded_sequence = encoded_sequence.detach().cpu().numpy()
                targets.extend(slided_targets)
                data.extend(encoded_sequence)

        targets = [targets[i] for i in backsort_inds]
        data = [data[i] for i in backsort_inds]

        targets_merged = []
        data_merged = []
        for slice_ in slices:
            slided_targets = np.vstack(targets[slice_])
            encoded_sequence = np.hstack(data[slice_])
            targets_merged.append(slided_targets)
            data_merged.append(encoded_sequence)

        all_decoded = []

        pb = tqdm(zip(data_merged, targets_merged), total=len(data_merged),
                  desc='[2/3 NerDetector] Decode found named entities')

        # TODO: all code below should be rewritten in the order of sanity
        for x, y in pb:
            decoded_ = []
            y = (softmax(y, 1) > self.threshold).argmax(1)
            y[0] = 0
            y[-1] = 0

            diffs = np.diff(y)
            s = list(np.where(diffs == 1)[0] + 1)
            e = list(np.where(diffs == -1)[0] + 1)
            if len(s) == len(e):
                pass
            elif (len(s) - len(e)) == 1:
                e.append(len(diffs))
            else:
                raise ValueError('Unhandled case')

            slices = [slice(s_, e_) for s_, e_ in zip(s, e)]

            for slice_ in slices:
                if slice_.stop - slice_.start >= self.min_span_len:
                    decoded_.append(self.tokenizer.decode(x[slice_], clean_up_tokenization_spaces=False))

            all_decoded.append(decoded_)

        final_texts = []

        pat = f'( *{self.token} *)+'
        pat = re.sub('\[', '\[', pat)
        pat = re.sub('\]', '\]', pat)

        pb = tqdm(zip(all_decoded, texts), total=len(all_decoded), desc='[3/3 NerDetector] Replace original text')
        for decoded, text in pb:
            if len(decoded) == 0:
                final_texts.append(text)
                continue

            spaceless_text = text.replace(' ', '')
            space_inds = []
            rep_mask = np.zeros(len(text))

            for i, m in enumerate(re.finditer('[^ ]+', text)):
                space_inds.extend(list(range(*m.span())))

            for d in decoded:
                d = d.replace(' ', '')
                for m in re.finditer(re.escape(d), spaceless_text):
                    orig_slice = slice(space_inds[m.span()[0]], space_inds[m.span()[1] - 1] + 1)
                    rep_mask[orig_slice] = 1

            rep_inds = np.where(rep_mask)[0]

            if sum(rep_inds) == 0:
                final_texts.append(text)
                continue

            slice_bounds = [0] + list(np.where(np.diff(rep_inds) != 1)[0] + 1) + [len(rep_inds)]
            final_slices = [slice(rep_inds[slice_bounds[i]], rep_inds[slice_bounds[i + 1] - 1] + 1) for i in
                            range(len(slice_bounds) - 1)]
            final_slices = sorted(final_slices, key=lambda x: x.start)

            final_text = text
            shift = 0
            for i, s in enumerate(final_slices):
                final_text = final_text[:s.start + shift] + self.token + final_text[s.stop + shift:]
                shift += len(self.token) - (s.stop - s.start)

            final_text = re.sub(pat, f' {self.token} ', final_text).strip()
            final_texts.append(final_text)

        return final_texts
