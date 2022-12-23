import numpy as np

from environment.anomaly_detection.abstract_preprocessor import AbstractPreprocessor
from environment.settings import MAX_ALLOWED_CORRELATION, DROP_TEMPORAL, DUPLICATE_HEADERS


class CorrelationPreprocessor(AbstractPreprocessor):
    def __init__(self):
        self.const_feats = None
        self.correlated_feats = None

    @staticmethod
    def get_constant_features(dataset):
        corr_const = dataset.corr()
        all_labels = set(corr_const.keys())
        # print("AD PRE: all", len(all_labels))

        corr_const.dropna(axis=1, how="all", inplace=True)
        corr_const.reset_index(drop=True)

        cropped_labels = set(corr_const.keys())
        # print("AD PRE: crop", len(cropped_labels))

        constant_feats = all_labels - cropped_labels
        # print("AD PRE: const", len(constant_feats))
        return constant_feats

    @staticmethod
    def get_highly_correlated_features(dataset):
        # https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
        corr_matrix = dataset.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_feats = [column for column in upper_tri.columns if any(upper_tri[column] > MAX_ALLOWED_CORRELATION)]
        # print("AD CORR:", len(correlated_feats))
        return correlated_feats

    def preprocess_dataset(self, dataset, is_normal=False):
        # Drop duplicated features
        dataset.drop(list(map(lambda header: header + ".1", DUPLICATE_HEADERS)),
                     inplace=True, axis=1)  # read_csv adds the .1

        # Drop temporal features
        dataset.drop(DROP_TEMPORAL, inplace=True, axis=1)

        # Remove vectors generated when the rasp did not have connectivity
        if len(dataset) > 1:  # avoid dropping single entries causing empty dataset
            dataset = dataset.loc[dataset["connectivity"] == 1]

        # Drop constant features
        if self.const_feats is None:  # must preprocess normal data first to align infected data to it
            self.const_feats = CorrelationPreprocessor.get_constant_features(dataset)
        dataset.drop(self.const_feats, inplace=True, axis=1)

        # Drop highly correlated features
        if self.correlated_feats is None:  # must preprocess normal data first to align infected data to it
            self.correlated_feats = CorrelationPreprocessor.get_highly_correlated_features(dataset)
        dataset.drop(self.correlated_feats, inplace=True, axis=1)

        # Reset index
        dataset.reset_index(inplace=True, drop=True)
        return dataset
