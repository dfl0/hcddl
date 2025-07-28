import os
import time
import timeit
import pickle
import concurrent

from logging import INFO, WARN
from typing import Optional

from flwr.common import (
    Code,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.server import Server
from flwr.server.history import History


class AsyncLocalParameterServer(Server):
    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        global_params_filepath = "params.pkl"
        log(INFO, f"Loading weights from file: {global_params_filepath}... ")
        with open(global_params_filepath, "rb") as file:
            parameters_ndarrays = pickle.load(file)
        log(INFO, "Done")
        parameters = ndarrays_to_parameters(parameters_ndarrays)

        return parameters

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()

        current_round = 0
        self.last_round = False

        update_count = 0

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        start_time = timeit.default_timer()
        while not self.last_round:
            log(INFO, "")
            global_params_filepath = "params.pkl"
            while True:
                if os.path.exists(global_params_filepath):
                    break
                log(INFO, "*** Waiting for global parameters ***")
                time.sleep(1)

            glb_sig_pipe = "glb_sig"

            if os.path.exists(glb_sig_pipe):
                with open(glb_sig_pipe, "r") as pipe:
                    signal = pipe.readline().strip()
                log(INFO, f"Signal received: {signal}")
                os.remove(glb_sig_pipe)
                if signal == "STOP":
                    break
                elif signal == "LAST":
                    self.last_round = True

            current_round += 1

            # Initialize parameters
            log(INFO, "[INIT]")
            self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

            # log(INFO, "Starting evaluation of global parameters")
            # res = self.strategy.evaluate(0, parameters=self.parameters)
            # if res is not None:
            #     log(
            #         INFO,
            #         "initial parameters (loss, other metrics): %s, %s",
            #         res[0],
            #         res[1],
            #     )
            #     history.add_loss_centralized(server_round=0, loss=res[0])
            #     history.add_metrics_centralized(server_round=0, metrics=res[1])
            # else:
            #     log(INFO, "Evaluation returned no results (`None`)")

            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            clients = self._client_manager.sample(2)
            fit_ins = FitIns(parameters=self.parameters, config={})

            futures = {executor.submit(client.fit, fit_ins, timeout=timeout, group_id=None): client for client in clients}
            for future in concurrent.futures.as_completed(futures):
                client = futures[future]
                try:
                    res_fit = future.result(timeout=None)  # TODO: determine what timout should be used
                    if res_fit is not None and res_fit.status.code == Code.OK:
                        gradients = res_fit.parameters

                        if gradients:
                            update_count += 1
                            log(INFO, f"Update {update_count} from client {client.cid}")

                            grads_filepath = f"grads_{update_count}.pkl"
                            with open(grads_filepath, "wb") as file:
                                updated_parameters_ndarrays = parameters_to_ndarrays(gradients)
                                pickle.dump(updated_parameters_ndarrays, file)
                            log(INFO, f"Gradients saved: {grads_filepath}")

                    else:
                        log(WARN, res_fit.status.message)

                except concurrent.futures.TimeoutError:
                    log(INFO, f"Client {client.cid} not done")
                except Exception as e:
                    log(INFO, f"Error with client {client.cid}: {e}")

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

        executor.shutdown(wait=True, cancel_futures=True)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed
