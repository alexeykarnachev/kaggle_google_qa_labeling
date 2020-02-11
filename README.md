# Kaggle Google QA Labeling Competition
## Here is solution of 41th place of the Kaggle Google QA Labeling competition

1. Install the package<br>
`pip install -e .`

2. Download NER-Model from Kaggle<br>
https://www.kaggle.com/alexeykarnachev/google-qa-ner

3. Prepare dataset<br>
`python scripts/prepare_dataset.py --seed=228 --train_df_file=data/google-quest-challenge/train.csv --test_df_file=data/google-quest-challenge/test.csv --tokenizer_cls=RobertaTokenizer --tokenizer_path=roberta-large --n_splits=7 --datasets_root=data/datasets/ --crop_strategies=both --dataset_cls=BiDataset --process_math --ner_model_dir=data/ner/code/bert_base_cased`<br><br>
--ner_model_dir is a path to downloaded NER model (from previous step)

4. Run experiment training<br>
`cd scripts`<br>
`python run_encoder_experiment.py --config_path=../configs/base_config.yaml`

5. Wait the training process end ...<br>
6. Archive the experiment directory<br>
`cd experiments`<br>
`tar zcvf <EXPERIMENT_DIR>.tar.gz <EXPERIMENT_DIR>`<br>
7. Send it to your kaggle datasets storage<br>
8. Now, you can inference the model in a kernel<br>
https://www.kaggle.com/alexeykarnachev/kernel1864bcfc13<br>
For this, attach the following datasets to the kernel:<br>
https://www.kaggle.com/alexeykarnachev/kaggle-google-qa-labeling (this package)<br>
https://www.kaggle.com/alexeykarnachev/google-qa-ner (NER model)<br>
https://www.kaggle.com/alexeykarnachev/transformersdependencies (transformers lib and dependencies)<br><br>
Also, attach trained experiment to the kernel<br><br>
Uncomment all lines in the kernel and replace the EXPERIMENT_NAME placeholder with your experiment name