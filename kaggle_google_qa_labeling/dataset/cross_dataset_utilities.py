import re
from typing import Any, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from kaggle_google_qa_labeling.utilities import dump_object

MATH_TOKEN = '[MATH]'
TITLE_SEP = '[TITLE_SEP]'
CODE_TOKEN = '[CODE]'
NEW_LINE = '\n'

additional_special_tokens = [TITLE_SEP, NEW_LINE, MATH_TOKEN, CODE_TOKEN]
REGULAR_MATH_PATTERN = re.compile(r'\$(?:.+?)\$', flags=re.U | re.I)
IRREGULAR_MATH_PATTERN = re.compile(r'\\begin{align}(.+?)\\end{align}', flags=re.U | re.I)


def get_pattern_spans(pattern: Any, text: str) -> List[Tuple[int, int]]:
    return [x.span() for x in pattern.finditer(text)]


def do_math_preprocessing(text: str) -> str:
    clean_text = REGULAR_MATH_PATTERN.sub(MATH_TOKEN, text)
    # some texts have irregular latex form
    clean_text = IRREGULAR_MATH_PATTERN.sub(MATH_TOKEN, clean_text)
    return clean_text


def get_tokenizer(tokenizer_cls, tokenizer_path):
    tokenizer: PreTrainedTokenizer = tokenizer_cls.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    return tokenizer


def get_tqa_codes(df, tokenizer, process_math: Optional[bool], ner_model):
    title_x = df['question_title'].values
    question_x = df['question_body'].values
    answer_x = df['answer'].values

    if ner_model:
        question_x, answer_x = ner_model(question_x), ner_model(answer_x)

    if process_math:
        question_x = [do_math_preprocessing(x) for x in question_x]
        answer_x = [do_math_preprocessing(x) for x in answer_x]

    title_x = [tokenizer.encode(x, add_special_tokens=False) for x in title_x]
    question_x = [tokenizer.encode(x, add_special_tokens=False) for x in question_x]
    answer_x = [tokenizer.encode(x, add_special_tokens=False) for x in answer_x]

    return title_x, question_x, answer_x


def get_crop_lengths(*arr_lengths, total_len):
    inp = np.array(arr_lengths)

    res = np.zeros_like(inp)

    if sum(inp) > total_len:
        while sum(res) < total_len:
            for i in range(len(inp)):
                if inp[i] > 0:
                    inp[i] -= 1
                    res[i] += 1
                    if sum(res) >= total_len:
                        break
    else:
        res = inp

    return res


def crop_sequence(x, seq_len, crop_strategy):
    if len(x) == seq_len:
        return x

    if crop_strategy == 'start':
        res = x[:seq_len]
    elif crop_strategy == 'end':
        res = x[-seq_len:]
    elif crop_strategy == 'middle':
        n_skip = len(x) - seq_len
        skip_left = n_skip // 2
        skip_right = n_skip - skip_left
        res = x[skip_right:(-skip_left)] if skip_left != 0 else x[skip_right:]
    elif crop_strategy == 'both':
        left_ids = -(seq_len - (seq_len // 2))
        res = x[:(seq_len // 2)] + (x[left_ids:] if left_ids != 0 else [])
    else:
        raise ValueError(f'Unknown crop strategy: {crop_strategy}')

    assert len(res) == seq_len

    return res


def get_crops(title_x, question_x, answer_x, max_len, tokenizer, crop_strategies):
    crop_x = {cs: [] for cs in crop_strategies}

    crop_strategies_pb = tqdm(crop_strategies, total=len(crop_strategies))

    for crop_strategy in crop_strategies_pb:
        crop_strategies_pb.set_description_str(crop_strategy)

        tokens_pb = tqdm(zip(title_x, question_x, answer_x), total=len(title_x), leave=False)
        for tx, qx, ax in tokens_pb:
            crop_lengths = get_crop_lengths(len(tx), len(qx), len(ax), total_len=max_len)
            tx_ = crop_sequence(tx, crop_lengths[0], crop_strategy=crop_strategy)
            qx_ = crop_sequence(qx, crop_lengths[1], crop_strategy=crop_strategy)
            ax_ = crop_sequence(ax, crop_lengths[2], crop_strategy=crop_strategy)

            x = join_sequences(tx_, qx_, ax_, tokenizer=tokenizer)
            crop_x[crop_strategy].append(x)

        crop_x[crop_strategy] = np.array(crop_x[crop_strategy])

    return crop_x


def get_train_d(train_index, crop_strategies, crop_x, features, targets, dataset_cls):
    train_targets = []
    train_data = []
    train_token_types = []
    train_F = [] if features is not None else None

    train_crop_x = {cs: crop_x[cs][train_index] for cs in crop_strategies}

    for i in range(len(train_index)):
        if train_crop_x[crop_strategies[0]][i][0] == train_crop_x[crop_strategies[-1]][i][0]:
            train_data.append([train_crop_x[crop_strategies[0]][i][0]])
            train_token_types.append([train_crop_x[crop_strategies[0]][i][1]])
            train_targets.append(targets[train_index[i]])
            if features is not None:
                train_F.append(features[train_index[i]])
        else:
            for cs in crop_strategies:
                train_data.append([train_crop_x[cs][i][0]])
                train_token_types.append([train_crop_x[cs][i][1]])
                train_targets.append(targets[train_index[i]])
                if features is not None:
                    train_F.append(features[train_index[i]])

    train_targets = np.array(train_targets)
    train_d = dataset_cls(train_data, train_F, train_token_types, train_targets, indexes=train_index)

    return train_d


def get_valid_d(valid_index, crop_strategies, crop_x, features, targets, dataset_cls):
    valid_data = []
    valid_token_types = []
    valid_F = [] if features is not None else None
    valid_targets = targets[valid_index] if targets is not None else None
    valid_crop_x = {cs: crop_x[cs][valid_index] for cs in crop_strategies}

    for i in range(len(valid_index)):
        valid_data.append([valid_crop_x[cs][i][0] for cs in crop_strategies])
        valid_token_types.append([valid_crop_x[cs][i][1] for cs in crop_strategies])

        if features is not None:
            valid_F.append(features[valid_index[i]])

    valid_d = dataset_cls(valid_data, valid_F, valid_token_types, valid_targets, indexes=valid_index)

    return valid_d


def prepare_crop_folds(crop_x, test_crop_x, extra_crop_x, extra_targets, train_features, test_features, n_splits,
                       targets, folds_dir, dataset_cls, groups, get_train_d_func, get_valid_d_func):

    crop_strategies = list(crop_x.keys())
    indices = range(len(crop_x[crop_strategies[0]]))

    gkf = GroupKFold(n_splits=n_splits).split(
        X=indices,
        groups=groups
    )

    if extra_crop_x is not None:
        extra_index = [-1] * len(extra_crop_x[crop_strategies[0]])
        extra_d = get_train_d_func(extra_index, crop_strategies, crop_x, extra_targets, dataset_cls)

    for i_fold, (train_index, valid_index) in enumerate(gkf):
        train_d = get_train_d_func(train_index, crop_strategies, crop_x, train_features, targets, dataset_cls)

        if extra_crop_x is not None:
            train_d.add_extra(extra_d)

        valid_d = get_valid_d_func(valid_index, crop_strategies, crop_x, train_features, targets, dataset_cls)

        fold_dir = folds_dir / str(i_fold)
        fold_dir.mkdir(parents=True, exist_ok=False)
        dump_object(train_d, fold_dir / 'train.pkl')
        dump_object(valid_d, fold_dir / 'valid.pkl')

    test_index = range(len(test_crop_x[crop_strategies[0]]))
    test_d = get_valid_d_func(test_index, crop_strategies, test_crop_x, test_features, None, dataset_cls)
    dump_object(test_d, folds_dir / 'test.pkl')

    all_index = range(len(crop_x[crop_strategies[0]]))
    all_d = get_train_d_func(all_index, crop_strategies, crop_x, train_features, targets, dataset_cls)
    dump_object(all_d, folds_dir / 'all.pkl')


def join_sequences(*inp_seqs, tokenizer: PreTrainedTokenizer):
    sep_id = tokenizer.convert_tokens_to_ids([TITLE_SEP])
    assert len(sep_id) == 1

    seq1 = inp_seqs[0] + sep_id + inp_seqs[1]
    seq2 = inp_seqs[2]
    res_x = tokenizer.build_inputs_with_special_tokens(seq1, seq2)
    res_t = [1] * (len(seq2) + 1)

    res_t = [0] * (len(res_x) - len(res_t)) + res_t

    assert len(res_x) == len(res_t)

    return res_x, res_t


def get_chunks(list_iterable, n):
    for i in range(0, len(list_iterable), n):
        yield list_iterable[i:i + n]


def flatten(list_iterable):
    return [item for sublist in list_iterable for item in sublist]


def pad_sequences(seqs, max_len, padding, pad_val):
    if not seqs:
        return np.array([])

    if padding not in ['post', 'pre']:
        raise ValueError("Padding must be 'post' or 'pre'")

    max_len_corrected = max([len(x) for x in seqs]) if not max_len else max_len

    new_seqs = []

    for seq in seqs:

        if padding == 'post':
            new_seq = seq[-max_len_corrected:]
            new_seq = new_seq + [pad_val] * (max_len_corrected - len(new_seq))
        else:
            new_seq = seq[:max_len_corrected]
            new_seq = [pad_val] * (max_len_corrected - len(new_seq)) + new_seq

        new_seqs.append(new_seq)

    return np.vstack(new_seqs)
