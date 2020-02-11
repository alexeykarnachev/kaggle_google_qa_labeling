import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent / '../'))

from kaggle_google_qa_labeling.dataset.bi_dataset import BiDataset
from kaggle_google_qa_labeling.dataset.multiclass_cross_dataset import MulticlassCrossDataset
from kaggle_google_qa_labeling.ner_detector import NERDetector
from kaggle_google_qa_labeling.factory import Factory
from kaggle_google_qa_labeling.dataset import cross_dataset_utilities
from kaggle_google_qa_labeling.dataset import bi_dataset_utilities
from kaggle_google_qa_labeling.dataset.cross_dataset import CrossDataset
from kaggle_google_qa_labeling.dataset import common_utilities

from kaggle_google_qa_labeling.utilities import get_cur_time_str, dump_json, seed_everything

parser = argparse.ArgumentParser(description='Prepare Dataset for Cross-models')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--train_df_file', type=Path,
                    help='input Data Frame file (splits will be prepared from this file)')  # TODO: ???
parser.add_argument('--test_df_file', type=Path, help='input Data Frame file (test file)')  # TODO: ???
parser.add_argument('--datasets_root', type=Path, help='root dir of all your datasets (subdir with new dataset '
                                                       'will be created there)')  # TODO: ???
parser.add_argument('--tokenizer_cls', type=str, help='tokenizer class name (from transformers library)')
parser.add_argument('--dataset_cls', type=str, help='dataset class name')
parser.add_argument('--tokenizer_path', type=str, help='tokenizer path (local path or remote name)')
parser.add_argument('--n_splits', type=int, help='Number of cross-val splits')
parser.add_argument('--crop_strategies', nargs='+', help='List crop strategies (start, end, middle, both)')
parser.add_argument('--ner_model_dir', required=False, type=Path,
                    help='Path to the pre-trained NER model dir')  # TODO: ???
parser.add_argument('--process_math', action='store_true', required=False,
                    help='Replace math expressions with a [MATH] token')
parser.add_argument('--extra_data_file', type=Path, required=False,
                    help='Additional data csv file with fields: '
                         'title, question_body, answer and all target columns')  # TODO: ???
parser.add_argument('--cat_features', nargs='+', help='List of categorical features', required=False)

args = parser.parse_args()

seed = args.seed
tokenizer_cls = Factory.get_class(f'transformers.{args.tokenizer_cls}')
datasets_root = args.datasets_root
tokenizer_path = args.tokenizer_path
train_df_file = args.train_df_file
test_df_file = args.test_df_file
n_splits = args.n_splits
crop_strategies = args.crop_strategies
dataset_cls_name = args.dataset_cls
cat_features = args.cat_features
ner_model_dir = args.ner_model_dir
process_math = args.process_math
extra_data_file = args.extra_data_file

cross_dataset_classes = {
    'CrossDataset': CrossDataset,
    'MulticlassCrossDataset': MulticlassCrossDataset
}

bi_dataset_classes = {
    'BiDataset': BiDataset
}

description = {'Dataset': args.__dict__}
description['Dataset']['crop_strategies'] = crop_strategies
description['Dataset']['cat_feature_dims'] = dict()

if __name__ == '__main__':
    seed_everything(seed)

    if ner_model_dir is not None:
        ner_model = NERDetector.from_model_dir(model_dir=ner_model_dir, device='cuda', bs=64,
                                               min_span_len=10, threshold=0.8)
    else:
        ner_model = None

    dataset_dir: Path = args.datasets_root / f'{dataset_cls_name}_{get_cur_time_str()}'
    folds_dir: Path = dataset_dir / 'folds'
    folds_dir.mkdir(exist_ok=False, parents=True)

    tokenizer = cross_dataset_utilities.get_tokenizer(tokenizer_cls, tokenizer_path)
    train = pd.read_csv(train_df_file)
    target_columns = train.columns[-30:]
    targets = train[target_columns].values
    test = pd.read_csv(test_df_file)

    # Encoding of categorical features
    if cat_features:
        df, test_df, cat_features_labels = common_utilities.categorize_features(train, test, cat_features)
        description['Dataset']['cat_feature_labels'] = cat_features_labels

        for col in cat_features:
            enc_dict = cat_features_labels[col]
            feature_dim = len(enc_dict) + 1
            description['Dataset']['cat_feature_dims'][col] = feature_dim

        train_features = df.loc[:, cat_features].values
        test_features = test_df.loc[:, cat_features].values
    else:
        train_features = None
        test_features = None

    title_x, question_x, answer_x = cross_dataset_utilities.get_tqa_codes(
        train, tokenizer=tokenizer, process_math=process_math, ner_model=ner_model
    )
    test_title_x, test_question_x, test_answer_x = cross_dataset_utilities.get_tqa_codes(
        test, tokenizer=tokenizer, process_math=process_math, ner_model=ner_model
    )

    if extra_data_file is not None:
        extra_df = pd.read_csv(extra_data_file)
        extra_title_x, extra_question_x, extra_answer_x = cross_dataset_utilities.get_tqa_codes(
            extra_df, tokenizer=tokenizer, process_math=process_math, ner_model=ner_model
        )
        extra_Y = train[target_columns].values
    else:
        extra_title_x = extra_question_x = extra_answer_x = extra_Y = None

    if dataset_cls_name in cross_dataset_classes:
        description['Dataset']['dataset_type'] = 'cross'
        dataset_cls = cross_dataset_classes[dataset_cls_name]
        max_len = tokenizer.max_len_sentences_pair - 1

        crop_x = cross_dataset_utilities.get_crops(title_x, question_x, answer_x, max_len, tokenizer, crop_strategies)
        test_crop_x = cross_dataset_utilities.get_crops(test_title_x, test_question_x, test_answer_x,
                                                        max_len, tokenizer, crop_strategies)
        if extra_data_file is not None:
            extra_crop_x = cross_dataset_utilities.get_crops(extra_title_x, extra_question_x, extra_answer_x,
                                                             max_len, tokenizer, crop_strategies)
        else:
            extra_crop_x = None

        cross_dataset_utilities.prepare_crop_folds(
            crop_x, test_crop_x, extra_crop_x, extra_Y, train_features, test_features, n_splits, targets, folds_dir,
            dataset_cls, train['question_body'], cross_dataset_utilities.get_train_d,
            cross_dataset_utilities.get_valid_d
        )
    elif dataset_cls_name in bi_dataset_classes:
        description['Dataset']['dataset_type'] = 'bi'
        dataset_cls = bi_dataset_classes[dataset_cls_name]
        max_len_tq = tokenizer.max_len_single_sentence - 1
        max_len_a = tokenizer.max_len_single_sentence

        crop_x = bi_dataset_utilities.get_crops(title_x, question_x, answer_x, (max_len_tq, max_len_a), tokenizer,
                                                crop_strategies)
        test_crop_x = bi_dataset_utilities.get_crops(test_title_x, test_question_x, test_answer_x,
                                                     (max_len_tq, max_len_a), tokenizer, crop_strategies)

        if extra_data_file is not None:
            extra_crop_x = bi_dataset_utilities.get_crops(extra_title_x, extra_question_x, extra_answer_x,
                                                          (max_len_tq, max_len_a), tokenizer, crop_strategies)
        else:
            extra_crop_x = None

        cross_dataset_utilities.prepare_crop_folds(
            crop_x, test_crop_x, extra_crop_x, extra_Y, train_features, test_features, n_splits, targets, folds_dir,
            dataset_cls, train['question_body'], bi_dataset_utilities.get_train_d, bi_dataset_utilities.get_valid_d
        )
    else:
        raise ValueError(f'Unknown dataset class name: {dataset_cls_name}')

    dump_json(description, dataset_dir / 'description.json')
