import sys
import argparse
from pathlib import Path

from torch.optim import Adam
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup, XLMRobertaModel

sys.path.append(str(Path(__file__).parent / '../'))

from kaggle_google_qa_labeling.dataset.multiclass_cross_dataset import MulticlassCrossDataset
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import get_tokenizer, additional_special_tokens
from kaggle_google_qa_labeling.models.utilities import resize_token_type_embeddings
from kaggle_google_qa_labeling.learner.learner import Learner
from kaggle_google_qa_labeling.factory import Factory

from kaggle_google_qa_labeling.utilities import load_json, load_object, seed_everything
from kaggle_google_qa_labeling.callbacks.callbacks import LRSchedulerCallback, ModelSaveCallback

parser = argparse.ArgumentParser(description='Fit Encoder-based model')
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--experiment_dir', type=Path, help='Existing experiment dir, to base training on')  # TODO: ???
parser.add_argument('--n_models', type=int, help='Number of models to train')
parser.add_argument('--device', type=str, help='Device to train on')

args = parser.parse_args()

seed = args.seed
experiment_dir = args.experiment_dir
n_models = args.n_models
device = args.device

cross_models = ['CrossEncoderModel.CrossEncoderModel']
bi_models = ['BiEncoderModel.BiEncoderModel']

if __name__ == '__main__':

    models_dir = experiment_dir / 'final_models'

    config = load_json(experiment_dir / 'description.json')
    data_config = config["Dataset"]
    config = config["Experiment"]
    config['device'] = device

    tokenizer_cls = Factory.get_class(f'transformers.{data_config["tokenizer_cls"]}')
    tokenizer_path = data_config["tokenizer_path"]
    tokenizer = get_tokenizer(tokenizer_cls, tokenizer_path)

    ds_path = Path(config['dataset_path']) / 'folds' / 'all.pkl'
    ds = load_object(ds_path)

    if isinstance(ds, MulticlassCrossDataset):
        y_dim = (30, 23)
    else:
        y_dim = 30

    dl = ds.get_data_loader(
        bs=config['fit_settings']['batch_size'],
        max_len=config['fit_settings']['max_len'],
        pad_id=tokenizer.pad_token_id,
        drop_last=True)

    loss = Factory.get_object(
        f"kaggle_google_qa_labeling.losses.{config['loss']}"
    )
    model_cls = Factory.get_class(f"kaggle_google_qa_labeling.models.{config['model']}")
    encoder_cls = Factory.get_class(config['backbone']['model'])

    for cur_model in range(n_models):
        print(f'Current model: {cur_model}')
        seed_everything(seed + cur_model)
        encoder = encoder_cls.from_pretrained(config['backbone']['pretrained'])
        if encoder_cls in [RobertaModel, XLMRobertaModel]:
            resize_token_type_embeddings(encoder, 2)

        encoder.resize_token_embeddings(len(additional_special_tokens) + encoder.config.vocab_size)

        # cat features logic
        if data_config.get('cat_features_dim'):

            min_ebm_dim = config['cat_features']['min_dim_size']
            emb_divisor = config['cat_features']['divisor']

            feature_dims = [d for _, d in data_config['cat_feature_dims'].items()]
            emb_feature_dims = [max(x / emb_divisor, min_ebm_dim) for x in feature_dims]
        else:
            feature_dims = None
            emb_feature_dims = None

        if config['model'] in bi_models:
            model = model_cls(
                q_encoder=encoder,
                a_encoder=encoder,
                feature_dims=feature_dims,
                emb_feature_dims=emb_feature_dims,
                mask_val=tokenizer.pad_token_id,
                y_dim=y_dim,
                loss_fn=loss,
                freeze_mode=config['backbone']['freeze_mode'],
                pooling=config['fit_settings']['pooling'],
                clf_hid_dim=config['fit_settings']['clf_hid_dim']
            ).to(config['device'])
        else:
            model = model_cls(
                encoder,
                mask_val=tokenizer.pad_token_id,
                y_dim=30,
                feature_dims=feature_dims,
                emb_feature_dims=emb_feature_dims,
                loss_fn=loss,
                freeze_mode=config['backbone']['freeze_mode'],
                pooling=config['fit_settings']['pooling'],
                clf_hid_dim=config['fit_settings']['clf_hid_dim']
            ).to(config['device'])

        optimizer_name = config['fit_settings']['optimizer'].lower()
        if optimizer_name == 'adamw':
            optimizer_cls = AdamW
        elif optimizer_name == 'adam':
            optimizer_cls = Adam
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        optimizer = optimizer_cls(model.parameters(), lr=float(config['fit_settings']['lr']))

        callbacks = [
            ModelSaveCallback(
                model_dir=models_dir / f'model_{cur_model}',
                save_each_epoch=False
            )
        ]

        warmup_steps = config['fit_settings']['warmup_steps']
        if warmup_steps > -1:
            callbacks.append(
                LRSchedulerCallback(
                    scheduler=get_linear_schedule_with_warmup(
                        optimizer,
                        int(warmup_steps),
                        (len(dl) * config['fit_settings']['epochs']) // config['fit_settings']['accum_steps']
                    ),
                    mode='step'
                )
            )

        learner = Learner(
            model=model,
            optimizer=optimizer,
            callbacks=callbacks,
            metrics=[
            ]
        )

        learner.fit(
            dl=(dl, None),
            n_epochs=config['fit_settings']['epochs'],
            accum_steps=config['fit_settings']['accum_steps'],
            eval_steps=config['fit_settings']['eval_steps'],
            use_all_gpu=False,
            fp16_opt_level=None,
            max_grad_norm=config['fit_settings']['max_grad_norm'],
            device=config['device']
        )
