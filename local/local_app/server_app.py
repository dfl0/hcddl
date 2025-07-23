from logging import INFO
from typing import List, Tuple, Optional, Union

import pickle

from flwr.common import (
    Context,
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log

from flwr.server import ServerAppComponents, ServerConfig, ServerApp
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from local_app.local_ps import LocalParameterServer  # , LocalParameterServerApp


class LocalFedAvg(FedAvg):
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"EdgeFedAvg(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        num_successful_clients = len(results)
        log(INFO, f"Aggregating gradients from {num_successful_clients} clients.")

        gradients_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        aggr_grads_filepath = "grads.pkl"
        with open(aggr_grads_filepath, "wb") as file:
            updated_parameters_ndarrays = parameters_to_ndarrays(gradients_aggregated)
            pickle.dump(updated_parameters_ndarrays, file)
        log(INFO, f"Aggregated complete, gradients saved: {aggr_grads_filepath}")

        return gradients_aggregated, metrics_aggregated


# custom (basic) weighted average
def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated_metrics: Metrics = {}
    total_examples = 0

    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        for metric_name, metric_value in client_metrics.items():
            aggregated_metrics[metric_name] = aggregated_metrics.get(metric_name, 0.0) + metric_value*num_examples

    for metric_name in aggregated_metrics:
        aggregated_metrics[metric_name] /= total_examples

    return aggregated_metrics


def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log(INFO, "Aggregating fit metrics from workers:")
    for i, (_, per_client_metrics) in enumerate(fit_metrics):
        log(INFO, f"  ({i}) {per_client_metrics.items()}")
    aggregated_metrics = weighted_avg(fit_metrics)
    return aggregated_metrics


def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log(INFO, "Aggregating evaluate metrics from workers:")
    for i, (_, per_client_metrics) in enumerate(eval_metrics):
        log(INFO, f"  ({i}) {per_client_metrics.items()}")
    aggregated_metrics = weighted_avg(eval_metrics)
    return aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    client_manager = SimpleClientManager()
    strategy = LocalFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        # fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=1)
    server = LocalParameterServer(client_manager=client_manager, strategy=strategy)

    return ServerAppComponents(server=server, config=config)


app = ServerApp(server_fn=server_fn)
