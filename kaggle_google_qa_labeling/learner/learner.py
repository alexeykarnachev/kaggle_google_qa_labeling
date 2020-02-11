import warnings
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


class Learner:
    def __init__(self, model, optimizer, callbacks, metrics):
        self.model = model
        self.optimizer = optimizer
        self.overall_step = 0
        self.callbacks = callbacks
        self.metrics = metrics

        self.n_epochs = 0
        self.cur_epoch = 0
        self.n_epoch_steps = 0
        self.cur_epoch_step = 0

        self.train_loss = 0
        self.valid_loss = 0

        self.Y_valid = []
        self.logits_valid = []

        self.device = None
        self.additional_log_fields = dict()

    @staticmethod
    def to_device(inp, device):

        if hasattr(inp, 'to'):
            inp = inp.to(device)
        else:
            for i in range(len(inp)):
                inp[i] = Learner.to_device(inp[i], device)

        return inp

    def fit(self, dl: Tuple[DataLoader, DataLoader], n_epochs, device, accum_steps, eval_steps, use_all_gpu,
            fp16_opt_level, max_grad_norm):

        self.device = device
        self.model = self.model.to(self.device)

        train_dl, valid_dl = dl

        n_gpu = torch.cuda.device_count() if use_all_gpu else 1

        if train_dl.batch_size / n_gpu != int(train_dl.batch_size / n_gpu):
            raise ValueError(f"You have {n_gpu} GPUs, batch size must be divisible by {n_gpu}")

        if fp16_opt_level is not None:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.n_epochs = n_epochs
        self.n_epoch_steps = len(train_dl)

        pb_epochs = tqdm(range(n_epochs), total=n_epochs, desc='Training')
        [c.on_train_start(learner=self) for c in self.callbacks]

        for cur_epoch in pb_epochs:
            self.cur_epoch = cur_epoch
            self.cur_epoch_step = 0
            pb_epochs.set_postfix({'Epoch': f'{cur_epoch + 1}/{n_epochs}'})
            pb_batches = tqdm(enumerate(train_dl), total=len(train_dl), desc='Epoch')

            for cur_batch, batch in pb_batches:
                self.cur_epoch_step += 1
                data, targets = self.to_device(batch[0], device), self.to_device(batch[1], device)

                self.model.train()

                loss, logits = self.model(data, targets)

                if n_gpu > 1:
                    loss = loss.mean()

                loss /= accum_steps

                if fp16_opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.overall_step += 1
                self.train_loss += loss.item()

                if self.overall_step % accum_steps == 0:

                    if fp16_opt_level is not None:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    [c.on_opt_step(learner=self) for c in self.callbacks]
                    self.optimizer.zero_grad()
                    pb_batches.set_postfix_str(self.get_log_str())
                    self.train_loss = 0

                if (self.overall_step % eval_steps == 0) and valid_dl is not None:
                    self.logits_valid, self.valid_loss, self.Y_valid = self.eval(dl=valid_dl)

                    if self.Y_valid is not None:
                        [m(logits=self.logits_valid, labels=self.Y_valid) for m in self.metrics]

                    [c.on_eval_end(learner=self) for c in self.callbacks]

            [c.on_epoch_end(learner=self) for c in self.callbacks]

        [c.on_train_end(learner=self) for c in self.callbacks]

    def eval(self, dl):
        device = next(self.model.parameters()).device

        with torch.no_grad():
            pb_valid = tqdm(dl, total=len(dl), desc='Validation')
            valid_loss = 0

            valid_predictions = []
            logits_valid = []

            for batch in pb_valid:
                data = self.to_device(batch[0], device)
                targets = self.to_device(batch[1], device) if len(batch) == 2 else None

                self.model.eval()
                loss, logits = self.model(data, targets)

                if targets is not None:
                    valid_loss += loss.item()

                    try:
                        targets = [targets[i].reshape(logits[i].shape) for i in range(len(logits))]
                    except:
                        warnings.warn("Can't reshape Y accordingly to logits. Skip reshaping!")

                    valid_predictions.append([list(x.detach().cpu().numpy()) for x in targets])

                logits_valid.append([list(x.detach().cpu().numpy()) for x in logits])

            logits_valid = [np.vstack(x) for x in list(zip(*logits_valid))]

            if targets is not None:
                valid_predictions = [np.vstack(x) for x in list(zip(*valid_predictions))]
                valid_loss /= len(pb_valid)
            else:
                valid_loss = np.nan
                valid_predictions = None

            return logits_valid, valid_loss, valid_predictions

    def get_log_str(self):
        epoch_str = f"Epoch: {self.cur_epoch + 1}/{self.n_epochs}"
        losses_str = f"Loss/Valid: {round(self.valid_loss, 5)}, Loss/Train: {round(self.train_loss, 5)}"
        metrics_str = ", ".join([f'{m.__repr__()}/Valid: {round(m.val, 5)}' for m in self.metrics])
        str_ = ', '.join([epoch_str, losses_str, metrics_str])

        if len(self.additional_log_fields):
            add_str = ", ".join(f'{k}: {v}' for k, v in self.additional_log_fields.items())
            str_ = ', '.join([str_, add_str])
        return str_
