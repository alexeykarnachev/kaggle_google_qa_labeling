from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from kaggle_google_qa_labeling.blend_utils import blend_sigmoids
from kaggle_google_qa_labeling.dataset.cross_dataset import CrossDataset
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import *
from kaggle_google_qa_labeling.evaluator.evaluator import Evaluator
from kaggle_google_qa_labeling.factory import Factory
from kaggle_google_qa_labeling.learner.learner import Learner


class CrossEncoderEvaluator(Evaluator):
    def __init__(self, experiment_dir: Path, blend_strategy, device, bs,
                 ignore_dir_names: List[str], process_math: Optional[bool], models_dir_name='models', ner_model=None):
        super().__init__(experiment_dir, blend_strategy=blend_strategy, models_dir_name=models_dir_name,
                         ignore_dir_names=ignore_dir_names)

        self.device = device
        self.bs = bs
        tokenizer_dir = Path(experiment_dir) / 'tokenizer'
        tokenizer_cls = Factory.get_class(f"transformers.{self.description['Dataset']['tokenizer_cls']}")

        self.tokenizer = tokenizer_cls.from_pretrained(str(tokenizer_dir))
        self.ner_model = ner_model
        self.process_math = process_math

    def _get_model(self, model_dir: Path) -> object:
        model_file = model_dir / 'model_final.pth'
        model = torch.load(model_file, map_location=self.device)
        return model

    def _get_data(self, inp_df: pd.DataFrame) -> DataLoader:

        max_len = self.description['Experiment']['fit_settings']['max_len']
        crop_strategies = self.description['Dataset']['crop_strategies']
        title_x, question_x, answer_x = get_tqa_codes(inp_df, self.tokenizer, ner_model=self.ner_model,
                                                      process_math=self.process_math)
        crop_x = get_crops(title_x, question_x, answer_x, max_len, self.tokenizer, crop_strategies)

        data = []
        token_types = []

        if self.description['Dataset'].get('cat_feature_labels'):
            has_cat_features = True
            cat_features = self.get_cat_feature_matrix(inp_df)
            F = []
        else:
            has_cat_features = False
            F = None

        for i in range(len(crop_x[crop_strategies[0]])):
            data.append([crop_x[cs][i][0] for cs in crop_strategies])
            token_types.append([crop_x[cs][i][1] for cs in crop_strategies])
            if has_cat_features:
                F.extend([cat_features[i] for _ in range(len(crop_strategies))])

        cross_dataset = CrossDataset(data, F, token_types, None)

        data_loader = cross_dataset.get_data_loader(
            bs=self.bs,
            max_len=max_len,
            pad_id=self.tokenizer.pad_token_id,
            drop_last=False,
            use_length_sampler=True
        )

        return data_loader

    def _evaluate(self, model: nn.Module, data: DataLoader) -> np.ndarray:
        learner = Learner(model, None, None, None)
        y_pred, _, _ = learner.eval(data)
        y_pred = blend_sigmoids(y_pred)
        backsort_inds = np.argsort(data.sampler.inds)
        y_pred = y_pred[backsort_inds]

        return y_pred
