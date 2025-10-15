import os
import re
import time
import pickle
import numpy as np

from logging import INFO

from flwr.client import NumPyClient
from flwr.common.logger import log


class AsyncLocalParameterServerClient(NumPyClient):
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        glb_sig_pipe = "glb_sig"

        if config["should_stop"]:
            log(INFO, "")
            log(INFO, "Sending signal `STOP`...")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("STOP\n")
            return parameters, 1, {}

        global_params_filepath = "params.pkl"
        with open(global_params_filepath, "wb") as file:
            pickle.dump(parameters, file)
        log(INFO, f"Global model parameters saved: {global_params_filepath}")

        with open(glb_sig_pipe, "w") as pipe:
            if config["last_round"]:
                log(INFO, "")
                log(INFO, "Sending signal `LAST`...")
                pipe.write("LAST\n")
            else:
                log(INFO, "")
                log(INFO, "Sending signal `GLB_PARAMS`...")
                pipe.write("GLB_PARAMS\n")

        log(INFO, "")
        log(INFO, "Waiting for more gradient updates")
        while True:
            grad_files = [f for f in os.listdir() if re.match(r"grads_\d+.pkl", f)]
            if grad_files:
                break
            time.sleep(1)

        log(INFO, f"Got gradient updates: {grad_files}")

        # calculate total avg computation time for workers since last time
        with open("comp_times.pkl", "rb") as file:
            worker_comp_times = pickle.load(file)
        print(worker_comp_times)

        total_avg_time = sum(worker_comp_times.values()) / len(worker_comp_times.items())
        print(f"Total avg time: {total_avg_time}")

        updated_grads_ndarrays = []
        for f in grad_files:
            log(INFO, f"Loading gradientss from file: {f}...")
            with open(f, "rb") as file:
                updated_grads_ndarrays.append(pickle.load(file))
            os.remove(f)

        avg_grads_ndarrays = [
            np.mean([gradients[i] for gradients in updated_grads_ndarrays], axis=0)
            for i in range(len(updated_grads_ndarrays[0]))
        ]

        return avg_grads_ndarrays, 1, {}

    # def evaluate(self, parameters, config):
    #     self.model.set_weights(parameters)
    #     loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    #     return 0.0, 1, {}
