import timeit
import tensorflow as tf
import concurrent

from logging import INFO, WARN
from typing import Optional

from flwr.common import (
    Parameters,
    FitIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Code,
)
from flwr.common.logger import log

from flwr.server.server import Server
from flwr.server.history import History

from .task import load_model, load_optimizer, load_loss_fn


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
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        max_clients = 1
        max_updates = 10

        update_count = 0

        start_time = timeit.default_timer()
        while update_count < max_updates:
            # sample between 1 and `max_clients` clients
            clients = self._client_manager.sample(max_clients, 1)
            fit_ins = FitIns(
                parameters=self.parameters,
                config={
                    "last_round": False,
                    "should_stop": False,
                }
            )

            futures = {executor.submit(client.fit, fit_ins, timeout=timeout, group_id=None): client for client in clients}
            for future in concurrent.futures.as_completed(futures):
                client = futures[future]
                try:
                    res_fit = future.result(timeout=timeout)  # TODO: determine what timout should be used
                    if res_fit is not None and res_fit.status.code == Code.OK:
                        gradients = res_fit.parameters

                        if gradients:
                            self._apply_gradients(gradients)
                            update_count += 1
                            log(INFO, f"Update {update_count}/{max_updates} from client {client.cid}")
                    else:
                        log(WARN, res_fit.status.message)
                except concurrent.futures.TimeoutError:
                    log(INFO, f"Client {client.cid} not done")
                except Exception as e:
                    log(INFO, f"Error with client {client.cid}: {e}")

            # TODO: stop training when target accuracy reached (like in sync. version)

        executor.shutdown(wait=True, cancel_futures=True)

        log(WARN, "Sending out last round of training")
        self.fit_round(
            server_round=-1,
            timeout=timeout,
        )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def _apply_gradients(self, gradients: Parameters):
        gradients_ndarrays = parameters_to_ndarrays(gradients)

        tf_gradients = [
            tf.convert_to_tensor(g, dtype=tf.float32) if g is not None else None
            for g in gradients_ndarrays
        ]

        self.strategy.optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_weights))
        self.parameters = ndarrays_to_parameters(self.model.get_weights())
