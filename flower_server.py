import flwr as fl
from flwr.server.strategy.fedopt import FedOpt
from src.fl.fl_utils import evaluate_metrics_aggregation_fn
from sklearn.metrics import roc_auc_score, log_loss
from omegaconf import OmegaConf

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    # evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, # for aggregated metrics
)
train_conf = OmegaConf.load("./train_conf.yaml")
# Choose number of rounds of training(One round consists of client side training, sending weights to server,
# aggregating weights at server and sending updated weights back to clients) and IP address to run server on.
temp = fl.server.start_server(
    server_address=train_conf.fl_server_port,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

# for num, df in temp.__dict__["metrics_distributed"]["cated_df"]:
#     print(round(roc_auc_score(df["label"], df["preds"]), 3))
