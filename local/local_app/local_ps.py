import os
import timeit
import pickle
from logging import INFO
from typing import Optional

from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
)

from flwr.common.logger import log
from flwr.server.server import Server
# from flwr.server.client_manager import ClientManager
from flwr.server.history import History
# from flwr.server.strategy import Strategy


class LocalParameterServer(Server):
    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        gbl_weights_filepath = "weights.pkl"
        log(INFO, f"Loading weights from file: {gbl_weights_filepath}... ")
        with open(gbl_weights_filepath, "rb") as file:
            parameters_ndarrays = pickle.load(file)
        log(INFO, "Done")
        parameters = ndarrays_to_parameters(parameters_ndarrays)

        log(INFO, "Removing weights file... ")
        os.remove(gbl_weights_filepath)
        log(INFO, "Done")

        return parameters

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()

        current_round = 0
        self.last_round = False

        start_time = timeit.default_timer()
        while not self.last_round:
            log(INFO, "")
            log(INFO, "*** Waiting for global parameters ***")
            log(INFO, "")
            glb_aggr_pipe = "glb_aggr_sig"
            if not os.path.exists(glb_aggr_pipe):
                os.mkfifo(glb_aggr_pipe)

            with open(glb_aggr_pipe, "r") as pipe:
                signal = pipe.readline().strip()  # wait for GLB_AGGR_W from Edge Aggregator client
            log(INFO, f"Signal received: {signal}")

            if signal == "STOP":
                os.remove(glb_aggr_pipe)
                break
            elif signal == "LAST":
                os.remove(glb_aggr_pipe)
                self.last_round = True

            current_round += 1

            # Initialize parameters
            log(INFO, "[INIT]")
            self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

            log(INFO, "Starting evaluation of global parameters")
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

            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )

            log(INFO, "Sending signal...")
            loc_aggr_pipe = "loc_aggr_sig"
            with open(loc_aggr_pipe, "w") as pipe:
                pipe.write("LOC_GRAD_W")  # -> Edge Aggregator client
            log(INFO, "Done")

            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                aggregated_gradients, fit_metrics, _ = res_fit

                if aggregated_gradients:
                    # aggregated_gradients_ndarrays = parameters_to_ndarrays(aggregated_gradients)

                    # tf_gradients = [
                    #     tf.convert_to_tensor(g, dtype=tf.float32) if g is not None else None
                    #     for g in aggregated_gradients_ndarrays
                    # ]

                    # self.strategy.optimizer.apply_gradients(zip(tf_gradients, self.model.trainable_weights))
                    # self.parameters = ndarrays_to_parameters(self.model.get_weights())

                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )

            # # Evaluate model using strategy implementation
            # res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            # if res_cen is not None:
            #     loss_cen, metrics_cen = res_cen
            #     log(
            #         INFO,
            #         "fit progress: (%s, %s, %s, %s)",
            #         current_round,
            #         loss_cen,
            #         metrics_cen,
            #         timeit.default_timer() - start_time,
            #     )
            #     history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            #     history.add_metrics_centralized(
            #         server_round=current_round, metrics=metrics_cen
            #     )

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

            # for line in io.StringIO(str(history)):
            #     log(INFO, "\t%s", line.strip("\n"))
            # log(INFO, "")

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed
