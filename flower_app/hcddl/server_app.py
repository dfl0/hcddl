import pickle
import os

from logging import INFO

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager

from hcddl.global_ps import GlobalParameterServer
from hcddl.async_global_ps import AsyncGlobalParameterServer
from hcddl.local_ps import LocalParameterServer
from hcddl.async_local_ps import AsyncLocalParameterServer

from hcddl.task import load_test_data, load_model, load_compiled_model, load_optimizer, load_loss_fn

class GlobalFedAvg(FedAvg):
    def __init__(self, *args, target_accuracy: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_accuracy = target_accuracy

        self.optimizer = load_optimizer()
        self.loss_fn = load_loss_fn()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"LocalFedAvg(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(self, *args, **kwargs):
        print("CONFIGURE FIT")
        client_ins = super().configure_fit(*args, **kwargs)
        print("DONE")
        return client_ins

    def aggregate_fit(self, *args, **kwargs):
        print("AGGREGATING PARAMETERS")
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(*args, **kwargs)
        print("DONE")
        return parameters_aggregated, metrics_aggregated


class LocalFedAvg(FedAvg):
    def __init__(self, *args, target_accuracy: float = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target_accuracy = target_accuracy

        self.optimizer = load_optimizer()
        self.loss_fn = load_loss_fn()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"GlobalFedAvg(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(self, *args, **kwargs):
        print("CONFIGURE FIT")
        client_ins = super().configure_fit(*args, **kwargs)
        print("DONE")
        return client_ins

    def aggregate_fit(self, server_round, results, failures):
        print("AGGREGATING GRADIENTS")
        gradients_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        print("DONE")

        print(os.listdir())

        aggr_grads_filepath = "grads.pkl"
        with open(aggr_grads_filepath, "wb") as file:
            updated_grads_ndarrays = parameters_to_ndarrays(gradients_aggregated)
            pickle.dump(updated_grads_ndarrays, file)
        log(INFO, f"Aggregation complete, gradients saved: {aggr_grads_filepath}")

        return gradients_aggregated, metrics_aggregated


def get_on_fit_config_fn(num_rounds):
    def fit_config(server_round):
        config = {
            "last_round": True if server_round >= num_rounds else False,
            "should_stop": True if server_round == -1 else False,
        }
        return config

    return fit_config


def evaluate_global(server_round, parameters, config):
    model = load_compiled_model()
    model.set_weights(parameters)

    x_test, y_test = load_test_data()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return loss, {"accuracy": accuracy}


def fit_metrics_aggregation_fn(fit_metrics):
    for i, (_, per_client_metrics) in enumerate(fit_metrics):
        log(INFO, f"    ({i}) {per_client_metrics.items()}")

    aggregated_metrics = {}
    total_examples = 0

    for num_examples, client_metrics in fit_metrics:
        total_examples += num_examples
        # accumulate weighted sum
        for key, val in client_metrics.items():
            aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + val * num_examples

    # compute weighted average
    for key in aggregated_metrics:
        aggregated_metrics[key] /= total_examples

    return aggregated_metrics


def server_fn(context):
    # Read from config
    arch = context.run_config["arch"]
    server_type = context.run_config["server-type"]
    aggr_type = context.run_config["aggr-type"]
    num_rounds = context.run_config["num-server-rounds"]
    target_accuracy = context.run_config["target-accuracy"]

    print("*"*50)
    print(arch)
    print(server_type)
    print(aggr_type)
    print("*"*50)

    parameters = ndarrays_to_parameters(load_model().get_weights())  # Get parameters to initialize global model

    client_manager = SimpleClientManager()

    if server_type == "global":
        strategy = GlobalFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            initial_parameters=parameters,
            on_fit_config_fn=get_on_fit_config_fn(num_rounds),
            evaluate_fn=evaluate_global,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            target_accuracy=target_accuracy,
        )

        server = (
            GlobalParameterServer(client_manager=client_manager, strategy=strategy) if aggr_type == "sync" else
            AsyncGlobalParameterServer(client_manager=client_manager, strategy=strategy)
        )

        config = ServerConfig(num_rounds=num_rounds)

    elif server_type == "local":
        strategy = LocalFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
            initial_parameters=parameters,
            on_fit_config_fn=get_on_fit_config_fn(num_rounds),
            evaluate_fn=evaluate_global,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            target_accuracy=target_accuracy,
        )

        server = (
            LocalParameterServer(client_manager=client_manager, strategy=strategy) if aggr_type == "sync" else
            AsyncLocalParameterServer(client_manager=client_manager, strategy=strategy)
        )

        config = ServerConfig(num_rounds=1)

    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
