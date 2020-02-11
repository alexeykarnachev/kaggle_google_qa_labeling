import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import crop_sequence, TITLE_SEP


def join_sequences(*inp_seqs, tokenizer: PreTrainedTokenizer):
    sep_id = tokenizer.convert_tokens_to_ids([TITLE_SEP])
    assert len(sep_id) == 1

    seq1 = inp_seqs[0] + sep_id + inp_seqs[1]
    seq2 = inp_seqs[2]

    res_x_tq = tokenizer.build_inputs_with_special_tokens(seq1)
    res_x_a = tokenizer.build_inputs_with_special_tokens(seq2)

    res_t_tq = [0] * len(res_x_tq)
    res_t_a = [1] * len(res_x_a)

    assert len(res_x_tq) == len(res_t_tq)
    assert len(res_x_a) == len(res_t_a)

    return (res_x_tq, res_x_a), (res_t_tq, res_t_a)


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


def get_crops(title_x, question_x, answer_x, max_len, tokenizer, crop_strategies):
    crop_x = {cs: [] for cs in crop_strategies}

    crop_strategies_pb = tqdm(crop_strategies, total=len(crop_strategies))

    for crop_strategy in crop_strategies_pb:
        crop_strategies_pb.set_description_str(crop_strategy)

        tokens_pb = tqdm(zip(title_x, question_x, answer_x), total=len(title_x), leave=False)
        for tx, qx, ax in tokens_pb:
            crop_lengths_tq = get_crop_lengths(len(tx), len(qx), total_len=max_len[0])
            crop_lengths_a = get_crop_lengths(len(ax), total_len=max_len[1])

            tx_ = crop_sequence(tx, crop_lengths_tq[0], crop_strategy=crop_strategy)
            qx_ = crop_sequence(qx, crop_lengths_tq[1], crop_strategy=crop_strategy)
            ax_ = crop_sequence(ax, crop_lengths_a[0], crop_strategy=crop_strategy)

            x = join_sequences(tx_, qx_, ax_, tokenizer=tokenizer)
            crop_x[crop_strategy].append(x)

        crop_x[crop_strategy] = np.array(crop_x[crop_strategy])

    return crop_x


def get_train_d(train_index, crop_strategies, crop_x, features, Y, dataset_cls):
    train_Y = []
    train_X_tq = []
    train_X_a = []
    train_T_tq = []
    train_T_a = []

    if features is not None:
        train_F = []
    else:
        train_F = None

    train_crop_x = {cs: crop_x[cs][train_index] for cs in crop_strategies}

    # (res_x_tq, res_x_a), (res_t_tq, res_t_a)
    for i in range(len(train_index)):
        if train_crop_x[crop_strategies[0]][i][0][0] == train_crop_x[crop_strategies[-1]][i][0][0]:
            train_X_tq.append([train_crop_x[crop_strategies[0]][i][0][0]])
            train_X_a.append([train_crop_x[crop_strategies[0]][i][0][1]])
            if features is not None:
                train_F.append(features[train_index[i]])
            train_T_tq.append([train_crop_x[crop_strategies[0]][i][1][0]])
            train_T_a.append([train_crop_x[crop_strategies[0]][i][1][1]])
            train_Y.append(Y[train_index[i]])
        else:
            for cs in crop_strategies:
                train_X_tq.append([train_crop_x[cs][i][0][0]])
                train_X_a.append([train_crop_x[cs][i][0][1]])
                if features is not None:
                    train_F.append(features[train_index[i]])
                train_T_tq.append([train_crop_x[cs][i][1][0]])
                train_T_a.append([train_crop_x[cs][i][1][1]])
                train_Y.append(Y[train_index[i]])

    train_Y = np.array(train_Y)
    train_d = dataset_cls(train_X_tq, train_X_a, train_F, train_T_tq, train_T_a, train_Y, indexes=train_index)

    return train_d


def get_valid_d(valid_index, crop_strategies, crop_x, features, Y, dataset_cls):
    valid_X_tq = []
    valid_X_a = []
    valid_T_tq = []
    valid_T_a = []

    if features is not None:
        valid_F = []
    else:
        valid_F = None

    valid_Y = Y[valid_index] if Y is not None else None
    valid_crop_x = {cs: crop_x[cs][valid_index] for cs in crop_strategies}

    for i in range(len(valid_index)):
        valid_X_tq.append([valid_crop_x[cs][i][0][0] for cs in crop_strategies])
        valid_X_a.append([valid_crop_x[cs][i][0][1] for cs in crop_strategies])
        if features is not None:
            valid_F.append(features[valid_index[i]])
        valid_T_tq.append([valid_crop_x[cs][i][1][0] for cs in crop_strategies])
        valid_T_a.append([valid_crop_x[cs][i][1][1] for cs in crop_strategies])

    valid_d = dataset_cls(valid_X_tq, valid_X_a, valid_F, valid_T_tq, valid_T_a, valid_Y, indexes=valid_index)

    return valid_d