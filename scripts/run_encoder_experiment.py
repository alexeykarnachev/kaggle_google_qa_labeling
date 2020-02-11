import sys
import argparse

from pathlib import Path

import yaml
import numpy as np
from torch.optim import Adam
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel

sys.path.append(str(Path(__file__).parent / '../'))

from kaggle_google_qa_labeling.factory import Factory
from kaggle_google_qa_labeling.learner.learner import Learner
from kaggle_google_qa_labeling.metrics.metrics import SpearmanMetric
from kaggle_google_qa_labeling.dataset.multiclass_cross_dataset import MulticlassCrossDataset
from kaggle_google_qa_labeling.models.utilities import resize_token_type_embeddings, get_params
from kaggle_google_qa_labeling.callbacks.callbacks import ModelSaveCallback, LRSchedulerCallback
from kaggle_google_qa_labeling.callbacks.freeze_encoder_on_plateau import FreezeEncoderOnPlateau
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import get_tokenizer, additional_special_tokens
from kaggle_google_qa_labeling.utilities import seed_everything, get_cur_time_str, load_json, dump_json, load_object

parser = argparse.ArgumentParser(description='Run Encoder-based model experiment')
parser.add_argument(
    '--config_path',
    type=str,
    help='Path to yaml/json config file'
)
args = parser.parse_args()

with open(args.config_path) as file:
    config = yaml.load(file)

if __name__ == '__main__':
    seed_everything(config['seed'])
    dataset_dir = Path(config['dataset_path'])
    experiments_root = Path('../experiments')
    experiment_dir = experiments_root / get_cur_time_str()
    models_dir = experiment_dir / 'fold_models'
    tokenizer_dir = experiment_dir / 'tokenizer'
    models_dir.mkdir(exist_ok=False, parents=True)
    tokenizer_dir.mkdir(exist_ok=False, parents=True)

    dataset_description = load_json(dataset_dir / 'description.json')
    tokenizer_cls = Factory.get_class(f'transformers.{dataset_description["Dataset"]["tokenizer_cls"]}')
    tokenizer_path = dataset_description["Dataset"]["tokenizer_path"]

    tokenizer = get_tokenizer(tokenizer_cls, tokenizer_path)

    loss = Factory.get_object(
        f"kaggle_google_qa_labeling.losses.{config['loss']}"
    )
    model_cls = Factory.get_class(f"kaggle_google_qa_labeling.models.{config['model']}")

    description = {
        "Experiment": config
    }
    description.update(dataset_description)

    dump_json(description, experiment_dir / 'description.json')
    tokenizer.save_pretrained(str(tokenizer_dir))

    oof_logits = []
    oof_y = []
    oof_metrics = []
    indexes = []

    for cur_fold in range(dataset_description['Dataset']['n_splits']):
        print(f'Current fold: {cur_fold}')
        fold_dir = dataset_dir / f'folds/{cur_fold}'

        ds_train = load_object(fold_dir / 'train.pkl')
        ds_valid = load_object(fold_dir / 'valid.pkl')

        if isinstance(ds_train, MulticlassCrossDataset):
            y_dim = (30, 23)
        else:
            y_dim = 30

        dl_train = ds_train.get_data_loader(
            bs=config['fit_settings']['batch_size'],
            max_len=config['fit_settings']['max_len'],
            pad_id=tokenizer.pad_token_id,
            drop_last=False)

        dl_valid = ds_valid.get_data_loader(
            bs=config['fit_settings']['batch_size'],
            max_len=config['fit_settings']['max_len'],
            pad_id=tokenizer.pad_token_id,
            drop_last=False)

        encoder_cls = Factory.get_class(config['backbone']['model'])
        encoder = encoder_cls.from_pretrained(config['backbone']['pretrained'], output_hidden_states=True)
        if encoder_cls in [RobertaModel]:
            resize_token_type_embeddings(encoder, 2)

        encoder.resize_token_embeddings(len(additional_special_tokens) + encoder.config.vocab_size)

        # cat features logic
        feature_dims = None
        emb_feature_dims = None

        if dataset_description["Dataset"].get('cat_feature_dims'):
            min_ebm_dim = config.get('cat_features', dict()).get('min_dim_size')
            emb_divisor = config.get('cat_features', dict()).get('divisor')

            if min_ebm_dim is not None and emb_divisor is not None:
                feature_dims = [d for _, d in dataset_description["Dataset"]['cat_feature_dims'].items()]
                emb_feature_dims = [int(max(x / emb_divisor, min_ebm_dim)) for x in feature_dims]

        model = model_cls(
            encoder,
            mask_val=tokenizer.pad_token_id,
            y_dim=y_dim,
            loss_fn=loss,
            pooling=config['fit_settings']['pooling'],
            clf_hid_dim=config['fit_settings']['clf_hid_dim'],
            dropout_rate=config['fit_settings']['dpt'],
            feature_dims=feature_dims,
            emb_feature_dims=emb_feature_dims
        ).to(config['device'])

        optimizer_name = config['fit_settings']['optimizer'].lower()
        if optimizer_name == 'adamw':
            optimizer_cls = AdamW
        elif optimizer_name == 'adam':
            optimizer_cls = Adam
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # ============================================================
        groups_lr = config['groups_lr']
        params = get_params(groups_lr, encoder, model)

        # ============================================================
        optimizer = optimizer_cls(params, lr=0)
        callbacks = [

        ]

        if config['save_fold_models']:
            callbacks.append(
                ModelSaveCallback(
                    model_dir=models_dir / f'model_{cur_fold}',
                    save_each_epoch=False
                )
            )

        warmup_steps = config['fit_settings']['warmup_steps']
        if warmup_steps > -1:
            callbacks.append(
                LRSchedulerCallback(
                    scheduler=get_linear_schedule_with_warmup(
                        optimizer,
                        int(warmup_steps),
                        (len(dl_train) * config['fit_settings']['epochs']) // config['fit_settings']['accum_steps']
                    ),
                    mode='step'
                )
            )

        freeze_encoder_on_plateau_patience = config['fit_settings']['freeze_encoder_on_plateau_patience'] or 0
        if freeze_encoder_on_plateau_patience > 0:
            callbacks.append(
                FreezeEncoderOnPlateau(freeze_encoder_on_plateau_patience)
            )

        learner = Learner(
            model=model,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=[
                SpearmanMetric()
            ]
        )

        if config['fit_settings']['eval_steps'] is None:
            config['fit_settings']['eval_steps'] = len(dl_train)

        learner.fit(
            dl=(dl_train, dl_valid),
            n_epochs=config['fit_settings']['epochs'],
            accum_steps=config['fit_settings']['accum_steps'],
            eval_steps=config['fit_settings']['eval_steps'],
            use_all_gpu=False,
            fp16_opt_level=None,
            max_grad_norm=config['fit_settings']['max_grad_norm'],
            device=config['device']
        )

        logits_valid, _, Y_valid = learner.eval(dl=dl_valid)
        backsort_inds = np.argsort(dl_valid.sampler.inds)
        logits_valid, Y_valid = logits_valid[0][backsort_inds], Y_valid[0][backsort_inds]

        oof_metrics.append(SpearmanMetric().calculate([logits_valid], [Y_valid]))
        oof_logits.append(logits_valid)
        oof_y.append(Y_valid)

        indexes.append(ds_valid.indexes)

    indexes = np.hstack(indexes)
    oof_logits = [np.concatenate(oof_logits, axis=0)]
    oof_y = [np.concatenate(oof_y, axis=0)]
    total_oof = SpearmanMetric().calculate(oof_logits, oof_y)

    np.save(str(experiment_dir / 'indexes'), indexes)
    np.save(str(experiment_dir / 'oof_logits'), oof_logits[0])
    np.save(str(experiment_dir / 'oof_y'), oof_y[0])

    metrics_res = {
        'each_oof': oof_metrics,
        'total_oof': total_oof
    }

    dump_json(metrics_res, experiment_dir / 'metrics.json')
