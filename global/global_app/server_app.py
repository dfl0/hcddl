import numpy as np
import keras

from functools import partial, reduce

from logging import INFO, WARN
from typing import List, Tuple, Optional, Union, Callable

from flwr.common import (
    Context,
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    NDArray,
    NDArrays,
    MetricsAggregationFn,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from .task import load_compiled_model, load_data, load_optimizer, load_loss_fn
from .global_ps import GlobalParameterServer


def get_on_fit_config_fn(num_rounds):
    def fit_config(server_round):
        config = {
            "last_round": True if server_round >= num_rounds else False,
            "should_stop": True if server_round == -1 else False,
        }
        return config

    return fit_config


global_test_samples = None
global_test_labels = None


def evaluate_global(server_round, parameters, config):
    model = load_compiled_model()
    model.set_weights(parameters)

    global global_test_samples
    global global_test_labels
    if global_test_samples is None or global_test_labels is None:
        _, (global_test_samples, global_test_labels) = load_data(0, 1)

    loss, accuracy = model.evaluate(global_test_samples, global_test_labels, verbose=0)
    return loss, {"accuracy": accuracy}


def aggregate_gradients_inplace(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    # Compute scaling factors for each result
    scaling_factors = np.asarray(
        [fit_res.num_examples / num_examples_total for _, fit_res in results]
    )

    def _try_inplace(
        x: NDArray, y: Union[NDArray, np.float64], np_binary_op: np.ufunc
    ) -> NDArray:
        return (  # type: ignore[no-any-return]
            np_binary_op(x, y, out=x)
            if np.can_cast(y, x.dtype, casting="same_kind")
            else np_binary_op(x, np.array(y, x.dtype), out=x)
        )

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        _try_inplace(x, scaling_factors[0], np_binary_op=np.multiply)
        for x in parameters_to_ndarrays(results[0][1].parameters)
    ]

    for i, (_, fit_res) in enumerate(results[1:], start=1):
        res = (
            _try_inplace(x, scaling_factors[i], np_binary_op=np.multiply)
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [
            reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
            for layer_updates in zip(params, res)
        ]

    return params


class GlobalFedAvg(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        target_accuracy: float = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

        self.target_accuracy = target_accuracy

        self.optimizer = load_optimizer()
        self.loss_fn = load_loss_fn()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CentralFedAvg(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        log(INFO, f"Aggregating gradients from {len(results)} clients.")

        # parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARN, "No fit_metrics_aggregation_fn provided")

        log(INFO, "Done")

        return parameters_aggregated, metrics_aggregated


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


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    target_accuracy = context.run_config["target-accuracy"]

    model = load_compiled_model()

    parameters = ndarrays_to_parameters(model.get_weights())

    client_manager = SimpleClientManager()
    strategy = strategy = GlobalFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=parameters,
        on_fit_config_fn=get_on_fit_config_fn(num_rounds),
        evaluate_fn=evaluate_global,
        target_accuracy=target_accuracy
    )
    config = ServerConfig(num_rounds=num_rounds)
    server = GlobalParameterServer(client_manager=client_manager, strategy=strategy)

    return ServerAppComponents(server=server, config=config)


app = ServerApp(server_fn=server_fn)
