import numpy as np

from kaggle_google_qa_labeling.callbacks.abstract_callback import AbstractCallback
from kaggle_google_qa_labeling.learner.learner import Learner


class FreezeEncoderOnPlateau(AbstractCallback):

    def __init__(self, patience: int):
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0
        self.done = False

    def on_epoch_end(self, learner: Learner):
        if not self.done:
            cur_loss = learner.valid_loss
            if cur_loss < self.min_loss:
                self.min_loss = cur_loss
                self.counter = 0
            else:
                self.counter += 1

            learner.additional_log_fields['EncoderFreezeAfter'] = self.patience - self.counter

            if self.counter == self.patience:
                for param in learner.model.encoder.parameters():
                    param.requires_grad = False

                self.done = True

                print('='*80 + '\n')
                print('Encoder has been FREEZED')
                print('=' * 80 + '\n')
