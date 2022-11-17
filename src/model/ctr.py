import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from ..train_prep.utils import split_patients, get_dense_sparse_features
from ..utils.constants import PLPConstants


def format_to_ctr(cov_pop):  # TODO : scaler 정리하기
    # feature_selection 들어가있으면...

    cov_pop.columns = cov_pop.columns.str.replace(
        ".", " ", regex=True
    )  # pytorch does not permit module name with "."

    for col in [col for col in cov_pop.columns if col not in PLPConstants.basics]:
        if int(cov_pop[col].sum()) == len(cov_pop[col]):
            cov_pop[col] = 0.0
    # substitute all 1.0 columns to 0.0 for DeepCTR embedding

    train_dt, valid_dt, test_dt = split_patients(cov_pop)
    dense_features, sparse_features = get_dense_sparse_features(cov_pop)

    fixlen_feature_columns = [
        SparseFeat(feat, 2)  # @note : 현재 categorical은 binary 변수밖에 못넣음
        for feat in sparse_features
    ] + [
        DenseFeat(
            feat,
            1,
        )
        for feat in dense_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train_label = np.array(train_dt["outcomeCount"])
    valid_label = np.array(valid_dt["outcomeCount"])
    test_label = np.array(test_dt["outcomeCount"])
    train_model_input = {name: train_dt[name] for name in feature_names}
    valid_model_input = {name: valid_dt[name] for name in feature_names}
    test_model_input = {name: test_dt[name] for name in feature_names}

    key_with_all_plus = [
        key
        for key in train_model_input
        if (int(train_model_input[key].sum()) == len(train_model_input[key]))
        | (int(valid_model_input[key].sum()) == len(valid_model_input[key]))
        | (int(test_model_input[key].sum()) == len(test_model_input[key]))
    ]
    for key in key_with_all_plus:
        train_model_input[key] = pd.Series(np.zeros_like(train_model_input[key]))
        valid_model_input[key] = pd.Series(np.zeros_like(valid_model_input[key]))
        test_model_input[key] = pd.Series(np.zeros_like(test_model_input[key]))

    return (
        train_model_input,
        valid_model_input,
        test_model_input,
        train_label,
        valid_label,
        test_label,
        linear_feature_columns,
        dnn_feature_columns,
    )


class CtrFeatureExtractor(FeatureExtractor):
    def __init__(self, app_dir, train_conf, client_conf):
        super().__init__(app_dir, train_conf, client_conf)

    def format_to_ctr(self, cov_pop, selected_features=None):
        # feature_selection 들어가있으면...

        cov_pop.columns = cov_pop.columns.str.replace(
            ".", " ", regex=True
        )  # pytorch does not permit module name with "."

        for col in [col for col in cov_pop.columns if col not in self.exclude_cols]:
            if int(cov_pop[col].sum()) == len(cov_pop[col]):
                cov_pop[col] = 0.0
        # substitute all 1.0 columns to 0.0 for DeepCTR embedding

        train_dt, valid_dt, test_dt = split_patients(cov_pop)
        dense_features, sparse_features = get_dense_sparse_features(cov_pop)
        if len(dense_features) != 0:
            mms = MinMaxScaler(feature_range=(0, 1))
            train_dt, valid_dt, test_dt = self.scaling(
                train_dt, valid_dt, test_dt, dense_features, mms
            )

        fixlen_feature_columns = [
            SparseFeat(feat, 2)
            # TODO : 현재 categorical은 binary 변수밖에 못넣음
            for feat in sparse_features
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in dense_features
        ]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        if selected_features is not None:  # selected features 정해져있으면...
            dnn_feature_columns = [
                i for i in dnn_feature_columns if i.name() in selected_features
            ]
            linear_feature_columns = [
                i for i in linear_feature_columns if i.name() in selected_features
            ]

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        train_label = np.array(train_dt["outcomeCount"])
        valid_label = np.array(valid_dt["outcomeCount"])
        test_label = np.array(test_dt["outcomeCount"])
        train_model_input = {name: train_dt[name] for name in feature_names}
        valid_model_input = {name: valid_dt[name] for name in feature_names}
        test_model_input = {name: test_dt[name] for name in feature_names}

        key_with_all_plus = [
            key
            for key in train_model_input
            if (int(train_model_input[key].sum()) == len(train_model_input[key]))
            | (int(valid_model_input[key].sum()) == len(valid_model_input[key]))
            | (int(test_model_input[key].sum()) == len(test_model_input[key]))
        ]
        for key in key_with_all_plus:
            train_model_input[key] = pd.Series(np.zeros_like(train_model_input[key]))
            valid_model_input[key] = pd.Series(np.zeros_like(valid_model_input[key]))
            test_model_input[key] = pd.Series(np.zeros_like(test_model_input[key]))

        return (
            train_model_input,
            valid_model_input,
            test_model_input,
            train_label,
            valid_label,
            test_label,
            linear_feature_columns,
            dnn_feature_columns,
        )

    def scaling(self, train_dt, valid_dt, test_dt, dense_features, mms):
        train_dt[dense_features] = mms.fit_transform(train_dt[dense_features])
        valid_dt[dense_features] = mms.transform(valid_dt[dense_features])
        test_dt[dense_features] = mms.transform(test_dt[dense_features])
        return train_dt, valid_dt, test_dt
