from pathlib import Path
from typing import Optional, List

import pandas as pd
from torch.utils.data.dataloader import DataLoader

from kaggle_google_qa_labeling.dataset.bi_dataset import BiDataset
from kaggle_google_qa_labeling.dataset.bi_dataset_utilities import *
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import get_tqa_codes
from kaggle_google_qa_labeling.evaluator.cross_encoder_evaluator import CrossEncoderEvaluator


class BiEncoderEvaluator(CrossEncoderEvaluator):
    def __init__(self, experiment_dir: Path, blend_strategy, device, bs,
                 ignore_dir_names: List[str], process_math: Optional[bool], models_dir_name='models', ner_model=None):
        super().__init__(experiment_dir, bs=bs, device=device, blend_strategy=blend_strategy,
                         models_dir_name=models_dir_name, ignore_dir_names=ignore_dir_names, process_math=process_math)

        self.ner_model = ner_model
        self.process_math = process_math

    def _get_data(self, inp_df: pd.DataFrame) -> DataLoader:
        max_len = self.description['Experiment']['fit_settings']['max_len']
        crop_strategies = self.description['Dataset']['crop_strategies']
        title_x, question_x, answer_x = get_tqa_codes(inp_df, self.tokenizer, ner_model=self.ner_model,
                                                      process_math=self.process_math)
        crop_x = get_crops(title_x, question_x, answer_x, (max_len - 1, max_len), self.tokenizer, crop_strategies)

        if self.description['Dataset'].get('cat_feature_labels'):
            has_cat_features = True
            cat_features = self.get_cat_feature_matrix(inp_df)
            F = []
        else:
            has_cat_features = False
            F = None

        X_tq = []
        X_a = []
        T_tq = []
        T_a = []

        for i in range(len(crop_x[crop_strategies[0]])):
            X_tq.append([crop_x[cs][i][0][0] for cs in crop_strategies])
            X_a.append([crop_x[cs][i][0][1] for cs in crop_strategies])
            T_tq.append([crop_x[cs][i][1][0] for cs in crop_strategies])
            T_a.append([crop_x[cs][i][1][1] for cs in crop_strategies])
            if has_cat_features:
                F.extend([cat_features[i] for _ in range(len(crop_strategies))])

        bi_dataset = BiDataset(X_tq, X_a, F, T_tq, T_a, None)

        data_loader = bi_dataset.get_data_loader(
            bs=self.bs,
            max_len=max_len,
            pad_id=self.tokenizer.pad_token_id,
            drop_last=False,
            use_length_sampler=True
        )

        return data_loader
