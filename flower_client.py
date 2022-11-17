import flwr as fl
from src.fl.fl_utils import TabularClient
from src.plp_extract.extract import extract
from src.train_prep.utils import split_patients, get_dense_sparse_features, df_to_loader
from omegaconf import OmegaConf
import warnings
import pandas as pd
from src.utils.misc import retry
from src.train_prep.dataset import TabularDataset

warnings.filterwarnings("ignore")


def main():
    """Create model, load data, define Flower client, start Flower client."""
    client_conf = OmegaConf.load("./client_conf.yaml")
    train_conf = OmegaConf.load("./train_conf.yaml")

    # build package
    extract(train_conf=train_conf, client_conf=client_conf)
    cov_pop = pd.read_parquet(f"./temps/cov_{train_conf.project}.parquet")

    # split data
    train, valid, test = split_patients(cov_pop, pct=[0.7, 0.1, 0.2])
    train, valid, test = (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
    train_loader, valid_loader, test_loader = df_to_loader(
        train, valid, test, TabularDataset, train_conf.batch_size
    )

    # Define client
    fl_client = TabularClient(
        train_loader,
        valid_loader,
        test_loader,
        client_conf=client_conf,
        train_conf=train_conf,
    )

    # try_client()
    fl.client.start_numpy_client(
        server_address=train_conf.fl_server_port, client=fl_client
    )


if __name__ == "__main__":
    main()
