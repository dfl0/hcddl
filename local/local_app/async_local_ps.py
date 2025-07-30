import os
import time
import timeit
import pickle
import concurrent

from logging import INFO, WARN
from typing import Optional, Tuple

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.server import Server
from flwr.server.history import History
from flwr.server.client_proxy import ClientProxy


class AsyncLocalParameterServer(Server):
    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()

        self.update_count = 0
        self.current_round = 0
        self.last_round = False

        log(INFO, "")
        log(INFO, "[INIT]")

        self.parameters = self._get_initial_parameters(0, timeout)

        start_time = timeit.default_timer()

        client_instructions = self.strategy.configure_fit(
            server_round=self.current_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        futures = {
            executor.submit(fit_client, client, ins, timeout)
            for client, ins in client_instructions
        }

        for future in futures:
            future.add_done_callback(
                lambda future: _handle_finished_future_after_fit(future, server=self, executor=executor, history=history),
            )

        while not self.last_round:
            self.current_round += 1

            log(INFO, "")
            log(INFO, "*** Waiting for new global parameters ***")
            log(INFO, "")

            glb_sig_pipe = "glb_sig"
            if not os.path.exists(glb_sig_pipe):
                os.mkfifo(glb_sig_pipe)

            with open(glb_sig_pipe, "r") as pipe:
                signal = pipe.readline().strip()
            log(INFO, f"Signal received: {signal}")
            os.remove(glb_sig_pipe)

            log(INFO, "")
            log(INFO, "[ROUND %s]", self.current_round)
            match signal:
                case "STOP":
                    break
                case "LAST":
                    self.last_round = True
                    pass
                case "GLB_PARAMS":
                    log(INFO, "Loading new global parameters")
                    self.parameters = self._get_initial_parameters(self.current_round, timeout)
                    pass
                case _:
                    log(WARN, f"Received unknown signal: {signal}")
                    pass

        executor.shutdown(wait=True, cancel_futures=True)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def get_client_fit_config(self):
        config = {}
        if self.strategy.on_fit_config_fn is not None:
            config = self.strategy.on_fit_config_fn(self.current_round)
        return config

    def _load_parameters(self):
        global_params_filepath = "params.pkl"
        log(INFO, f"Loading weights from file: {global_params_filepath}... ")
        with open(global_params_filepath, "rb") as file:
            parameters_ndarrays = pickle.load(file)
        log(INFO, "Done")
        parameters = ndarrays_to_parameters(parameters_ndarrays)

        return parameters

    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        global_params_filepath = "params.pkl"

        if server_round == 0:
            while True:
                if os.path.exists(global_params_filepath):
                    break
                log(INFO, "*** Waiting for global parameters ***")
                time.sleep(1)

        parameters = self._load_parameters()

        return parameters


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int = None
) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,
    server: AsyncLocalParameterServer,
    executor: concurrent.futures.ThreadPoolExecutor,
    history: History,
) -> None:
    try:
        client, res_fit = future.result()  # TODO: determine what timout should be used

        if res_fit is not None and res_fit.status.code == Code.OK:
            gradients = res_fit.parameters

            if gradients:
                server.update_count += 1
                log(INFO, f"Update {server.update_count} from client {client.cid}")

                grads_filepath = f"grads_{server.update_count}.pkl"
                with open(grads_filepath, "wb") as file:
                    updated_parameters_ndarrays = parameters_to_ndarrays(gradients)
                    pickle.dump(updated_parameters_ndarrays, file)
                log(INFO, f"Gradients saved: {grads_filepath}")

        else:
            log(WARN, res_fit.status.message)

        new_ins = FitIns(server.parameters, config=server.get_client_fit_config())
        future = executor.submit(fit_client, client, new_ins, timeout=None)
        future.add_done_callback(lambda future: _handle_finished_future_after_fit(future, server, executor, history))

    except concurrent.futures.TimeoutError:
        log(WARN, "Client not done")
    except Exception as e:
        log(WARN, e)
