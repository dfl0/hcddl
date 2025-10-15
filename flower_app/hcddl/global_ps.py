import timeit
import threading
import tensorflow as tf
from logging import INFO, WARN
from typing import Optional
from datetime import datetime

from flwr.common import Scalar, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log

from flwr.server.server import Server, FitResultsAndFailures, fit_clients
from flwr.server.history import History

from .task import load_model


class GlobalParameterServer(Server):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = load_model()
        self.t_round = []
        self.t_conf = []
        self.t_fit = []
        self.t_aggr = []
        self.t_comp = []

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        history = History()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

        max_accuracy = 0.0

        latest_loss_cen = None
        latest_accuracy_cen = None

        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
            latest_loss_cen = res[0]
            if "accuracy" in res[1]:
                latest_accuracy_cen = res[1]["accuracy"]
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):

            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            round_start_time = timeit.default_timer()

            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )

            # TODO: fix having to change between gradients/parameters manually

            if res_fit is not None:
                # aggregated_gradients, fit_metrics, _ = res_fit
                # if aggregated_gradients:
                #     self._apply_gradients(aggregated_gradients)

                # history.add_metrics_distributed_fit(
                #     server_round=current_round, metrics=fit_metrics
                # )

                aggregated_parameters, fit_metrics, _ = res_fit
                if aggregated_parameters:
                    self.parameters = aggregated_parameters
                    self.model.set_weights(parameters_to_ndarrays(aggregated_parameters))

                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            elapsed_round = timeit.default_timer() - round_start_time

            self.t_round.append((current_round, elapsed_round))

            metrics_cen = {"round_time": elapsed_round}
            history.add_metrics_centralized(
                server_round=current_round, metrics=metrics_cen
            )

            # TODO: add ability to pass a custom number of rounds to wait before evaluating again (currently 10)

            if current_round % 10 == 0:
                # Evaluate model using strategy implementation
                res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    latest_loss_cen = loss_cen

                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )

                    history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                    history.add_metrics_centralized(
                        server_round=current_round, metrics=metrics_cen
                    )

                    if self.strategy.target_accuracy is not None:
                        if "accuracy" in metrics_cen:
                            log(INFO, "Global accuracy: %.4f (Target: %.4f)", metrics_cen["accuracy"], self.strategy.target_accuracy)

                            latest_accuracy_cen = metrics_cen["accuracy"]
                            max_accuracy = max(latest_accuracy_cen, max_accuracy)

                            if metrics_cen["accuracy"] >= self.strategy.target_accuracy:
                                log(INFO, "Target accuracy reached, stopping training")

                                log(WARN, "Sending out last round of training")
                                self.fit_round(
                                    server_round=-1,
                                    timeout=timeout,
                                )
                                break
                        else:
                            log(WARN, "Evaluation function did not return an 'accuracy' metric")
                    else:
                        log(WARN, "No target accuracy provided")

            if current_round % 10 == 0:
                log(INFO, "Starting async evaluation")
                threading.Thread(
                    target=async_eval,
                    args=(self.strategy, current_round, self.parameters, history),
                    daemon=True  # won't block shutdown
                ).start()

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # training done
        end_time = timeit.default_timer()
        elapsed = end_time - start_time

        # Evaluate model using strategy implementation
        res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            latest_loss_cen = loss_cen

            log(
                INFO,
                "fit progress: (%s, %s, %s)",
                current_round,
                loss_cen,
                metrics_cen,
            )

            history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            history.add_metrics_centralized(
                server_round=current_round, metrics=metrics_cen
            )

            if "accuracy" in metrics_cen:
                log(INFO, "Global accuracy: %.4f (Target: %.4f)", metrics_cen["accuracy"], self.strategy.target_accuracy)
                latest_accuracy_cen = metrics_cen["accuracy"]
                max_accuracy = max(latest_accuracy_cen, max_accuracy)

        save_results(self, elapsed, num_rounds, latest_loss_cen, latest_accuracy_cen, max_accuracy)

        return history, elapsed

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
    ]:
        print(f"{timestamp()}  BEFORE CONFIGURE_FIT")
        t_start = timeit.default_timer()
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        t_elapsed = timeit.default_timer() - t_start
        print(f"{timestamp()}  AFTER CONFIGURE_FIT")
        self.t_conf.append((server_round, t_elapsed))

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        print(f"{timestamp()}  BEFORE PRINTING NUM AVAILABLE")
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        print(f"{timestamp()}  BEFORE FIT_CLIENTS")
        t_start = timeit.default_timer()
        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        t_elapsed = timeit.default_timer() - t_start
        print(f"{timestamp()}  AFTER FIT_CLIENTS")
        self.t_fit.append((server_round, t_elapsed))

        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        print(f"{timestamp()}  BEFORE AGGREGATE_FIT")
        t_start = timeit.default_timer()
        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        t_elapsed = timeit.default_timer() - t_start
        print(f"{timestamp()}  AFTER AGGREGATE_FIT")
        self.t_aggr.append((server_round, t_elapsed))

        parameters_aggregated, metrics_aggregated = aggregated_result
        self.t_comp.append((server_round, metrics_aggregated['comp_time']))

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def _apply_gradients(self, aggregated_gradients: Parameters):
        aggregated_gradients_ndarrays = parameters_to_ndarrays(aggregated_gradients)

        tf_gradients = [
            tf.convert_to_tensor(g, dtype=tf.float32) if g is not None else None
            for g in aggregated_gradients_ndarrays
        ]

        self.strategy.optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_weights))
        self.parameters = ndarrays_to_parameters(self.model.get_weights())


def async_eval(strategy, round_num, parameters, history):
    """Run evaluation in a background thread."""
    res_cen = strategy.evaluate(round_num, parameters=parameters)
    if res_cen is not None:
        loss_cen, metrics_cen = res_cen
        history.add_loss_centralized(server_round=round_num, loss=loss_cen)
        history.add_metrics_centralized(
            server_round=round_num, metrics=metrics_cen
        )
        log(INFO, "[ASYNC EVAL] Round %s - Loss: %.4f, Metrics: %s", round_num, loss_cen, metrics_cen)
    else:
        log(INFO, "[ASYNC EVAL] Round %s - No results", round_num)

def timestamp():
    return datetime.now().strftime('%H:%M:%S.%f')

def save_results(
        server: GlobalParameterServer,
        total_time: float,
        total_steps: int,
        final_loss: float,
        final_accuracy: float,
        max_accuracy: float
):
    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    with open("results.txt", "a") as f:
        f.write(f"[ Results for run {datetime_str} ]\n")
        f.write(f"Total time:\t{total_time}\n")
        f.write(f"Total steps:\t{total_steps}\n")
        f.write(f"Final loss:\t{final_loss}\n")
        f.write(f"Final accuracy:\t{final_accuracy}\n")
        f.write(f"Max accuracy:\t{max_accuracy}\n")
        f.write(f"(total) t_conf:\t{sum(t for _, t in server.t_conf)}\n")
        f.write(f"(total) t_round:\t{sum(t for _, t in server.t_round)}\n")
        f.write(f"(total) t_fit:\t{sum(t for _, t in server.t_fit)}\n")
        f.write(f"(total) t_aggr:\t{sum(t for _, t in server.t_aggr)}\n")
        f.write(f"(total) t_comp:\t{sum(t for _, t in server.t_comp)}\n\n")

    print()
    print(f"[ Results for run {datetime_str} ]")
    print(f"Total time: {total_time}")
    print(f"Total steps: {total_steps}")
    print(f"Final loss: {final_loss}")
    print(f"Final accuracy: {final_accuracy}")
    print(f"Max accuracy: {max_accuracy}")
    print()
    print(f"(total) t_conf:\t{sum(t for _, t in server.t_conf)}")
    print(f"(total) t_round:\t{sum(t for _, t in server.t_round)}")
    print(f"(total) t_fit:\t{sum(t for _, t in server.t_fit)}")
    print(f"(total) t_aggr:\t{sum(t for _, t in server.t_aggr)}")
    print(f"(total) t_comp:\t{sum(t for _, t in server.t_comp)}")
    print()
