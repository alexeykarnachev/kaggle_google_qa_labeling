import pandas as pd
from sklearn.preprocessing import LabelEncoder


def categorize_features(
        train : pd.DataFrame,
        test: pd.DataFrame,
        features: list):

    encoder_dicts = {}
    for col in features:
        encoder = LabelEncoder()
        encoder.fit(train[col])
        encoder_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))
        encoder_dicts[col] = encoder_dict

        train[col] = train[col].map(encoder_dict)
        test[col] = test[col].apply(lambda x: encoder_dict.get(x, 0))

    return train, test, encoder_dicts