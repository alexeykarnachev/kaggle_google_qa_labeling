from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from kaggle_google_qa_labeling.blend_utils import blend_ranks, blend_sigmoids, blend_mean
from kaggle_google_qa_labeling.utilities import load_json


class Evaluator(ABC):
    def __init__(self, experiment_dir: Path, blend_strategy, models_dir_name, ignore_dir_names: List[str] = None):
        """
        :param experiment_dir: path to the experiment. Experiment dir must contain description.json and also,
            models directory with models folders inside
        """
        self.experiment_dir = Path(experiment_dir)
        self.description = load_json(self.experiment_dir / 'description.json')
        self.model_dirs = list((self.experiment_dir / models_dir_name).iterdir())
        self.blend_strategy = blend_strategy
        self.ignore_dir_names = ignore_dir_names

    def get_cat_feature_matrix(self, df):
        cat_features_labels = self.description['Dataset']['cat_feature_labels'].keys()

        features_to_take = []
        for col in cat_features_labels.keys():
            df[col] = df[col].apply(lambda x: cat_features_labels[col].get(x, 0))
            features_to_take.append(col)

        return df.loc[:, features_to_take].values

    @abstractmethod
    def _get_model(self, model_dir: Path) -> object:
        pass

    @abstractmethod
    def _get_data(self, inp_df: pd.DataFrame) -> object:
        pass

    @abstractmethod
    def _evaluate(self, model: object, data: object) -> np.ndarray:
        pass

    def run(self, inp_df: pd.DataFrame) -> np.ndarray:
        data = self._get_data(inp_df)

        predictions = []

        for model_dir in self.model_dirs:
            if model_dir.name in self.ignore_dir_names:
                continue
            model = self._get_model(model_dir)
            model_predictions = self._evaluate(model, data)
            predictions.append(model_predictions)

        if self.blend_strategy == 'rank':
            final_prediction = blend_ranks(predictions)
        elif self.blend_strategy == 'sigmoid':
            final_prediction = blend_sigmoids(predictions)
        elif self.blend_strategy == 'mean':
            final_prediction = blend_mean(predictions)
        else:
            raise ValueError(f'Unknown blend strategy: {self.blend_strategy}')

        return final_prediction
