import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
import pickle
import traceback
from flwr.common import ndarrays_to_parameters
from pathlib import Path
from ..utils.constants import PLPConstants
from ..train_prep.utils import get_dense_sparse_features, split_patients_outcome
from ..model.get_model import get_model
from ..train_prep.pl_module import PL
import pytorch_lightning as pl


class TabularClient(fl.client.NumPyClient):
    def __init__(
        self, train_loader, valid_loader, test_loader, train_conf=None, client_conf=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = pl.Trainer(
            max_epochs=1,
            amp_backend="native",
            amp_level="O2",
            num_sanity_val_steps=0,
        )  # ㅇ
        self.model = PL(
            network=get_model(train_conf.model)(),
            batch_size=train_conf.batch_size,
            loss=torch.nn.BCELoss(),
            train_loader=train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
        )
        self.train_n = len(train_loader.dataset)
        self.valid_n = len(valid_loader.dataset)
        # dataset, dataloader 제작하기

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict)
        except:
            pass  # TODO : key matching에 에러가 존재함...model network layer 이름의 문제일지도...? 다른 모델에서는 이런 문제 생긴적 없음.

    def fit(self, parameters, config=None):

        self.set_parameters(parameters)
        self.trainer.fit(model=self.model)
        return self.get_parameters(), self.train_n, {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.trainer.test()

        loss = 0.5  # log_loss(valid_y, preds)
        auroc = 0.5  # round(roc_auc_score(valid_y, preds), 4)
        df = None  # pd.DataFrame({"preds": preds, "label": valid_y})

        return loss, self.valid_n, {"Loss": float(round(loss, 4)), "AUROC": auroc}

    def save_initial(self):
        with open("./temps/initial_parameter.pkl", "wb") as f:
            pickle.dump(ndarrays_to_parameters(self.get_parameters()), f)


def evaluate_metrics_aggregation_fn(
    fit_metrics: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """Aggregate metrics from multiple clients."""
    try:
        num_sum = 0
        auroc_weighted_sum = 0
        loss_weighted_sum = 0

        for num, metrics in fit_metrics:
            num_sum += num
            # auroc_weighted_sum += metrics["AUROC"] * num
            loss_weighted_sum += metrics["Loss"] * num

        # metrics["weight_auroc"] = auroc_weighted_sum / num_sum
        metrics["weight_loss"] = loss_weighted_sum / num_sum
    except Exception:
        traceback.print_exc()
    return metrics


def plp_formatting(cov_pop, params):
    cov_ref_dict = {
        **dict(
            zip(
                params["cov_ref"]["covariateName"],
                params["cov_ref"]["covariateId"].astype(int).astype(str),
            )
        )
    }

    cov_pop.columns = cov_pop.columns.map(lambda x: cov_ref_dict.get(x, 0))
    cov = cov_pop.drop([0], axis=1)

    # selected된 컬럼만 남기고 지워버림
    cov_selected = cov[params["selected_features"].astype(str).tolist()]
    cov_selected = cov_selected.loc[:, ~cov_selected.columns.duplicated()]
    cov_pop = pd.concat([cov_pop[PLPConstants.basics_outcome], cov_selected], axis=1)
    return cov_pop


def client_fn(cid: str, cov_pops, hospitals, params, initial_weights=None):
    """Create a Flower client representing a single organization."""
    hos = hospitals[int(cid)]

    cov_pop = plp_formatting(cov_pops, hos, params)

    train, valid, test = split_patients_outcome(
        cov_pop, random_state=params["random_state"]
    )  #
    dense_features, sparse_features = get_dense_sparse_features(
        cov_pop, except_cols=PLPConstants.basics_outcome
    )

    return TabularClient(
        train,
        valid,
        test,
        dense_features,
        sparse_features,
        params,
        initial_weights=initial_weights,
    )
