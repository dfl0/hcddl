import timeit
import tensorflow as tf
import concurrent
from datetime import datetime

from logging import INFO, WARN
from typing import Optional, Union

from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Code,
)
from flwr.common.logger import log

from flwr.server.server import Server
from flwr.server.history import History
from flwr.server.client_proxy import ClientProxy

from .task import load_model


class AsyncGlobalParameterServer(Server):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = load_model()
        self.model.compile(
            optimizer=self.strategy.optimizer,
            loss=self.strategy.loss_fn
        )

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        history = History()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

        max_updates = num_rounds

        update_count = 0
        max_accuracy = 0.0

        # latest_loss_cen = None
        # latest_accuracy_cen = None
        latest_loss_cen = 0
        latest_accuracy_cen = 0

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

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        start_time = timeit.default_timer()
        while update_count < max_updates:
            client_instructions = self.strategy.configure_fit(
                server_round=update_count + 1,
                parameters=self.parameters,
                client_manager=self._client_manager,
            )

            futures = {
                executor.submit(fit_client, client, ins, timeout)
                for client, ins in client_instructions
            }

            fut_results = []
            fut_failures = []

            for future in concurrent.futures.as_completed(futures):
                try:
                    client, res_fit = future.result()

                    if res_fit is not None and res_fit.status.code == Code.OK:
                        _handle_finished_future_after_fit(
                            future=future, results=fut_results, failures=fut_failures
                        )

                        gradients = res_fit.parameters
                        loc_ps_metrics = res_fit.metrics
                        history.add_metrics_distributed_fit(
                            update_count, metrics=loc_ps_metrics
                        )

                        if gradients:
                            self._apply_gradients(gradients)
                            update_count += 1
                            log(INFO, "")
                            log(INFO, f"Received aggregated gradients from Local Parameter Server {
                                client.cid}")
                            log(INFO, f"Update {update_count}/{max_updates}")

                    else:
                        log(WARN, res_fit.status.message)

                except concurrent.futures.TimeoutError:
                    log(INFO, f"Client {client.cid} not done")

                except Exception as e:
                    log(INFO, f"Error with client {client.cid}: {e}")

            num_updates = len(fut_results)

            # Aggregate Local Parameter Server results
            log(INFO, "Aggregating Local PS results...")
            grads_aggregated, metrics_aggregated = self.strategy.aggregate_fit(num_updates, fut_results, fut_failures)
            history.add_metrics_centralized(update_count, metrics=metrics_aggregated)

            # Evaluate global model every X updates
            if update_count % 5 == 0:
                log(INFO, "Evaluating current gobal parameters")
                res_cen = self.strategy.evaluate(update_count, parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    latest_loss_cen = loss_cen

                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        update_count,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )

                    history.add_loss_centralized(server_round=update_count, loss=loss_cen)
                    history.add_metrics_centralized(server_round=update_count, metrics=metrics_cen)

                    if self.strategy.target_accuracy is not None:
                        if "accuracy" in metrics_cen:
                            log(INFO, "Global accuracy: %.4f (Target: %.4f)",
                                latest_accuracy_cen, self.strategy.target_accuracy)

                            latest_accuracy_cen = metrics_cen["accuracy"]
                            max_accuracy = max(latest_accuracy_cen, max_accuracy)

                            if latest_accuracy_cen >= self.strategy.target_accuracy:
                                log(INFO, "Target accuracy reached, stopping training...")
                                log(WARN, "Sending out termination(?) round")
                                self.fit_round(
                                    server_round=-1,
                                    timeout=timeout
                                )
                                break

                        else:
                            log(WARN, "Evaluation function did not return an 'accuracy' metric")

                    else:
                        log(WARN, "No target accuracy provided")

        # calculate total time taken for traininn
        elapsed = timeit.default_timer() - start_time

        log(INFO, "Shutting down executor...")
        executor.shutdown(wait=True)

        save_results(self, elapsed, update_count, latest_loss_cen, latest_accuracy_cen, max_accuracy)

        return history, elapsed

    def _apply_gradients(self, gradients: Parameters):
        gradients_ndarrays = parameters_to_ndarrays(gradients)

        tf_gradients = [
            tf.convert_to_tensor(g, dtype=tf.float32) if g is not None else None
            for g in gradients_ndarrays
        ]

        self.strategy.optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_weights))
        self.parameters = ndarrays_to_parameters(self.model.get_weights())


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int = None
) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[ClientProxy, FitRes]],
    failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def save_results(
        server: AsyncGlobalParameterServer,
        total_time: float,
        total_steps: int,
        final_loss: float,
        final_accuracy: float,
        max_accuracy: float
):
    datetime_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    with open("results.txt", "a") as f:
        f.write(f"[ Results for run {datetime_str} ]\n")
        f.write(f"Total time: {total_time}\n")
        f.write(f"Total steps: {total_steps}\n")
        f.write(f"Final loss: {final_loss}\n")
        f.write(f"Final accuracy: {final_accuracy}\n")
        f.write(f"Max accuracy: {max_accuracy}\n\n")
