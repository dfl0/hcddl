import timeit
import tensorflow as tf
from logging import INFO, WARN
from typing import Optional

from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log

from flwr.server.server import Server
from flwr.server.history import History

from .task import load_model


class GlobalParameterServer(Server):
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

        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):

            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )

            if res_fit is not None:
                aggregated_gradients, fit_metrics, _ = res_fit
                if aggregated_gradients:
                    self._apply_gradients(aggregated_gradients)

                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
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

            # # Evaluate model on a sample of available clients
            # res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            # if res_fed is not None:
            #     loss_fed, evaluate_metrics_fed, _ = res_fed
            #     if loss_fed is not None:
            #         history.add_loss_distributed(
            #             server_round=current_round, loss=loss_fed
            #         )
            #         history.add_metrics_distributed(
            #             server_round=current_round, metrics=evaluate_metrics_fed
            #         )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def _apply_gradients(self, aggregated_gradients: Parameters):
        aggregated_gradients_ndarrays = parameters_to_ndarrays(aggregated_gradients)

        tf_gradients = [
            tf.convert_to_tensor(g, dtype=tf.float32) if g is not None else None
            for g in aggregated_gradients_ndarrays
        ]

        self.strategy.optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_weights))
        self.parameters = ndarrays_to_parameters(self.model.get_weights())
