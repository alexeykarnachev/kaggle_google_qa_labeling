import logging
from pathlib import Path
from typing import Callable, Optional, Dict

import torch
from tqdm.auto import tqdm
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from kaggle_google_qa_labeling.callbacks.abstract_callback import AbstractCallback
from kaggle_google_qa_labeling.learner.learner import Learner
from kaggle_google_qa_labeling.utilities import sigmoid, flatten

import numpy as np


class FileLoggerCallback(AbstractCallback):

    def __init__(self, file_path: Path, debug: bool = False):
        """
        :param file_path:
        :param debug: bool, if True, datetime will be not logged (useful for testing purposes)
        """

        Path(file_path.parent).mkdir(exist_ok=True, parents=True)

        if debug:
            logging.basicConfig(filename=file_path, filemode='w', format='[DEBUG] %(message)s', level=logging.INFO)
        else:
            logging.basicConfig(filename=file_path, filemode='w',
                                format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def on_eval_end(self, learner: Learner):
        str_ = self.get_log_str(learner) + '\n'
        self.logger.info(str_)

    @staticmethod
    def get_log_str(learner: Learner):
        epoch_str = f"Epoch: {learner.cur_epoch + 1}/{learner.n_epochs}, "
        step_str = f"Step: {learner.cur_epoch_step + 1}/{learner.n_epoch_steps}, "
        losses_str = f"Loss/Valid: {round(learner.valid_loss, 5)}, Loss/Train: {round(learner.train_loss, 5)}, "
        metrics_str = ", ".join([f'{m.__repr__()}/Valid: {round(m.val, 5)}' for m in learner.metrics])
        str_ = epoch_str + step_str + losses_str + metrics_str
        return str_


class LRSchedulerCallback(AbstractCallback):

    def __init__(self, scheduler, mode: str, get_step_kwargs: Optional[Callable[[Learner], Dict]] = None):
        self.scheduler = scheduler
        self.get_step_kwargs = get_step_kwargs
        self.mode = mode

    def on_opt_step(self, learner: Learner):
        if self.mode == 'step':
            self.do_step(learner=learner)

    def on_eval_end(self, learner: Learner):
        if self.mode == 'eval':
            self.do_step(learner=learner)

    def do_step(self, learner: Learner):
        if self.get_step_kwargs is not None:
            kwargs = self.get_step_kwargs(learner)
        else:
            kwargs = dict()
        self.scheduler.step(**kwargs)


class ModelSaveCallback(AbstractCallback):

    def __init__(self, model_dir: Path, save_each_epoch: bool):
        self.model_dir = model_dir
        self.save_each_epoch = save_each_epoch

        Path(self.model_dir).mkdir(exist_ok=True, parents=True)

    @staticmethod
    def save_model(model, model_file: Path):
        torch.save(model, str(model_file))

    @staticmethod
    def get_file_name(learner: Learner):
        return f"model_{int(learner.cur_epoch)}.pth"

    def on_train_end(self, learner: Learner):
        torch.save(learner.model, self.model_dir / "model_final.pth")

    def on_epoch_end(self, learner: Learner):
        if self.save_each_epoch:
            torch.save(learner.model, self.model_dir / self.get_file_name(learner))


class TensorboardWriterCallback(AbstractCallback):
    def __init__(self, tb_writer: SummaryWriter, description: str):
        self.tb_writer = tb_writer
        self.tb_writer.add_text('Description', description)

    def on_eval_end(self, learner: Learner):
        if learner.Y_valid is not None:
            for valid_metric in learner.metrics:
                self.tb_writer.add_scalar(f'{valid_metric.__repr__()}/valid', valid_metric.val, learner.overall_step)

            self.tb_writer.add_pr_curve(
                'PRCurve/valid',
                labels=np.array(learner.Y_valid[0]),
                predictions=sigmoid(np.array(learner.logits_valid[0])),
                global_step=learner.overall_step
            )

    def on_opt_step(self, learner: Learner):
        self.tb_writer.add_scalar(f'Loss/train', learner.train_loss, learner.overall_step)
        self.tb_writer.add_scalar(f'Loss/valid', learner.valid_loss, learner.overall_step)

        lr = learner.optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar(f'Learning-Rate', lr, learner.overall_step)


class MRRScoreTBCallback(AbstractCallback):
    """
    Callback for Bi/Poly-Models. It calculates MRR-score and also select top-n candidates for each context sample
    """

    def __init__(self, dl_holdout, tb_writer, device):
        self.dl_holdout = dl_holdout
        self.tb_writer = tb_writer
        self.device = device

    def on_epoch_end(self, learner):
        with torch.no_grad():
            model = learner.model
            model.eval()

            h_context = []
            h_candidate = []

            pb_h = tqdm(self.dl_holdout, total=len(self.dl_holdout), desc='MRR-Score: Calculate H')
            for batch in pb_h:
                data, targets = learner.to_device(batch[0], self.device), learner.to_device(batch[1], self.device)
                T_context, X_context, T_candidate, X_candidate = data

                h_context_batch = model.get_h_context(X_context, T_context).detach().cpu().numpy()
                h_candidate_batch = model.get_h_candidate(X_candidate, T_candidate).detach().cpu().numpy()

                h_context.append(h_context_batch)
                h_candidate.append(h_candidate_batch)

            h_context = flatten(h_context)
            h_candidate = np.vstack(h_candidate)

            dl_h_candidate = DataLoader(
                dataset=TensorDataset(torch.FloatTensor(h_candidate)),
                batch_size=self.dl_holdout.batch_size,
                shuffle=False
            )

            all_contexts_logits = []

            pb_h_context = tqdm(h_context, total=len(h_context), desc='Scoring')

            for h_context in pb_h_context:
                context_logits = []
                h_context_batch = torch.FloatTensor([h_context]).to(self.device)

                for h_candidate_batch in dl_h_candidate:
                    h_candidate_batch = h_candidate_batch[0].to(self.device)
                    context_logits_ = model.get_logits(h_context_batch, h_candidate_batch).detach().cpu().numpy()
                    context_logits.extend(context_logits_)

                all_contexts_logits.append(context_logits)

            all_contexts_logits = np.array(all_contexts_logits)
            ranks = np.abs(np.apply_along_axis(rankdata, 1, all_contexts_logits) - all_contexts_logits.shape[1]) + 1
            mrr = np.mean(1 / ranks.diagonal())

            self.tb_writer.add_scalar(f'MRRScore/valid', mrr, learner.cur_epoch)