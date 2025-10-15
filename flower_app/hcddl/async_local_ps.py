import os
import time
import timeit
import pickle
import concurrent
import threading

from logging import INFO, WARN
from typing import Optional, Union

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    EvaluateRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.server import Server
from flwr.server.history import History
from flwr.server.client_proxy import ClientProxy


class AsyncLocalParameterServer(Server):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.active_futures = set()
        self.future_lock = threading.Lock()

        self.worker_comp_times = {}

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

        fut_results = []
        fut_failures = []

        with self.future_lock:
            self.active_futures.update(futures)

        for future in futures:
            future.add_done_callback(
                lambda future: _handle_finished_future_after_fit(
                    future=future, results=fut_results, failures=fut_failures, server=self, executor=executor, history=history
                ),
            )

        while not self.last_round:
            with self.future_lock:
                """
                In case `fut_results` is updated again before aggregation is
                  complete - waiting for aggregation to finish to release the
                  lock could take longer, preventing workers from continuing
                  as soon as possible)
                """
                curr_fut_results = fut_results.copy()
                curr_fut_failures = fut_failures.copy()

            # Aggregate Worker results/metrics
            if curr_fut_results:
                log(INFO, "")
                log(INFO, f"Aggregating Worker results for parameters No. {self.current_round}...")
                grads_aggregated, metrics_aggregated = self.strategy.aggregate_fit(self.current_round, curr_fut_results, curr_fut_failures)
                history.add_metrics_distributed_fit(self.current_round, metrics=metrics_aggregated)

            with self.future_lock:
                fut_results = fut_results[len(curr_fut_results):]
                fut_failures = fut_failures[len(curr_fut_failures):]

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

        log(INFO, "[SHUTDOWN] Waiting for in-progress client updates to finish...")
        while True:
            with self.future_lock:
                if not self.active_futures:
                    break
            time.sleep(1)
        log(INFO, "[SHUTDOWN] All client updates complete.")

        executor.shutdown(wait=True)

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
    results: list[tuple[ClientProxy, EvaluateRes]],
    failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    server: AsyncLocalParameterServer,
    executor: concurrent.futures.ThreadPoolExecutor,
    history: History,
) -> None:
    with server.future_lock:
        server.active_futures.discard(future)

    failure = future.exception()
    if failure is not None:
        failures.append(failure)

    try:
        result = future.result()  # TODO: determine what timout should be used
        client, res_fit = result

        if res_fit is not None and res_fit.status.code == Code.OK:
            gradients = res_fit.parameters
            worker_metrics = res_fit.metrics
            history.add_metrics_distributed_fit(server.current_round, metrics=worker_metrics)

            server.worker_comp_times[client.cid] = server.worker_comp_times.get(client.cid, 0) + worker_metrics["comp_time"]

            comp_times_filepath = "comp_times.pkl"
            with open(comp_times_filepath, "wb") as file:
                pickle.dump(server.worker_comp_times, file)
            log(INFO, f"Computation times saved: {comp_times_filepath}")

            if gradients:
                with server.future_lock:
                    server.update_count += 1
                    update_count = server.update_count

                log(INFO, f"Update {update_count} from client {client.cid}")

                grads_filepath = f"grads_{update_count}.pkl"
                with open(grads_filepath, "wb") as file:
                    updated_parameters_ndarrays = parameters_to_ndarrays(gradients)
                    pickle.dump(updated_parameters_ndarrays, file)
                log(INFO, f"Gradients saved: {grads_filepath}")

                results.append(result)

        else:
            log(WARN, res_fit.status.message)

        if not server.last_round:
            new_ins = FitIns(server.parameters, config=server.get_client_fit_config())
            new_future = executor.submit(fit_client, client, new_ins, timeout=None)
            with server.future_lock:
                server.active_futures.add(new_future)
            new_future.add_done_callback(
                lambda future: _handle_finished_future_after_fit(future, results, failures, server, executor, history)
            )

    except concurrent.futures.TimeoutError:
        log(WARN, "Client not done")

    except Exception as e:
        # should only ever be any other exceptions, not one in the future as that is already handled
        log(WARN, e)
