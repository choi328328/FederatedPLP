from loguru import logger
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

basics = [
    "rowId",
    "subjectId",
    "cohortStartDate",
    "cohortId",
    "ageYear",
    "gender",
    "outcomeCount",
    "timeAtRisk",
    "survivalTime",
]


def get_cov_pops(cov_pop, imp, common_cols):

    feature_sorted = (
        pd.DataFrame(imp.values()).fillna(0).mean().sort_values(ascending=False)
    )
    selected_feat = feature_sorted.index.tolist()
    selected_cols = cov_pop.columns[cov_pop.columns.isin(selected_feat)]
    selected_cols = selected_cols[selected_cols.isin(common_cols)].tolist()[:1000]
    cov_pop = cov_pop[basics + selected_cols]
    return cov_pop


def df_to_loader(train, valid, test, dataset_class, batch_size, pin_memory=True):
    train_dataset, valid_dataset, test_dataset = (
        dataset_class(train),
        dataset_class(valid),
        dataset_class(test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=32,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=32,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def split_patients(cov_pop, pct=None):
    "split dataset by patients while preserving the outcome ratio"
    if pct is None:
        pct = [0.7, 0.1, 0.2]

    train_pat, test_valid = train_test_split(
        cov_pop.subjectId.unique(), test_size=(1 - pct[0]), random_state=328
    )
    valid_pat, test_pat = train_test_split(
        test_valid, test_size=pct[2] / (pct[1] + pct[2]), random_state=328
    )
    train, valid, test = (
        cov_pop[cov_pop.subjectId.isin(train_pat)],
        cov_pop[cov_pop.subjectId.isin(valid_pat)],
        cov_pop[cov_pop.subjectId.isin(test_pat)],
    )

    return train, valid, test


def split_patients_outcome(cov_pop, pct=None, random_state=328):
    "split dataset by patients while preserving the outcome ratio"

    case_df = cov_pop.query("outcomeCount==1")
    cont_df = cov_pop.query("outcomeCount==0")

    case_df_pid = case_df.subjectId.unique()
    case_tr_pid, case_vd_pid = train_test_split(
        case_df_pid, test_size=0.3, random_state=random_state
    )
    case_vd_pid, case_te_pid = train_test_split(
        case_vd_pid, test_size=0.66, random_state=random_state
    )

    case_tr_df = case_df[case_df.subjectId.isin(case_tr_pid)]
    case_vd_df = case_df[case_df.subjectId.isin(case_vd_pid)]
    case_te_df = case_df[case_df.subjectId.isin(case_te_pid)]

    cont_in_case_tr = cont_df[cont_df.subjectId.isin(case_tr_pid)]
    cont_in_case_vd = cont_df[cont_df.subjectId.isin(case_vd_pid)]
    cont_in_case_te = cont_df[cont_df.subjectId.isin(case_te_pid)]

    cont_df_pid = cont_df[~cont_df.subjectId.isin(case_df.subjectId)].subjectId.unique()

    cont_tr_pid, cont_vd_pid = train_test_split(
        cont_df_pid, test_size=0.3, random_state=random_state
    )
    cont_vd_pid, cont_te_pid = train_test_split(
        cont_vd_pid, test_size=0.66, random_state=random_state
    )

    cont_tr_df = cont_df[cont_df.subjectId.isin(cont_tr_pid)]
    cont_vd_df = cont_df[cont_df.subjectId.isin(cont_vd_pid)]
    cont_te_df = cont_df[cont_df.subjectId.isin(cont_te_pid)]

    train = pd.concat([case_tr_df, cont_tr_df, cont_in_case_tr])
    valid = pd.concat([case_vd_df, cont_vd_df, cont_in_case_vd])
    test = pd.concat([case_te_df, cont_te_df, cont_in_case_te])
    return train, valid, test


def get_dense_sparse_features(cov_pop):

    cov_pop_ex = cov_pop.drop(basics, axis=1)
    unique_values = cov_pop_ex.nunique()
    sparse_features = cov_pop_ex.loc[:, unique_values <= 2].columns.tolist()
    dense_features = cov_pop_ex.loc[:, unique_values > 2].columns.tolist()
    return dense_features, sparse_features
