import os
import re
import time
import pickle
import numpy as np

from logging import INFO

from flwr.common.logger import log
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context


class LocalParameterServerClient(NumPyClient):
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        glb_sig_pipe = "glb_sig"

        if config["should_stop"]:
            log(INFO, "Sending signal... ")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("STOP\n")  # -> Local PS
            log(INFO, "Done")
            return parameters, 1, {}
        elif config["last_round"]:
            log(INFO, "Sending signal... ")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("LAST\n")  # -> Local PS
            log(INFO, "Done")

        global_params_filepath = "params.pkl"
        with open(global_params_filepath, "wb") as file:
            pickle.dump(parameters, file)
        log(INFO, f"Global model params saved: {global_params_filepath}")

        log(INFO, "Sending signal...")
        glb_sig_pipe = "glb_sig"
        with open(glb_sig_pipe, "w") as pipe:
            pipe.write("GLB_AGGR_W\n")  # -> Local PS
        log(INFO, "Done")

        log(INFO, "Waiting for local training and aggregation to finish...")

        loc_sig_pipe = "loc_sig"
        if not os.path.exists(loc_sig_pipe):
            os.mkfifo(loc_sig_pipe)

        with open(loc_sig_pipe, "r") as pipe:
            signal = pipe.readline().strip()  # wait for LOC_GRAD_W from Local PS
        log(INFO, f"Signal received: {signal}")
        os.remove(loc_sig_pipe)

        updated_grads_filepath = "grads.pkl"
        log(INFO, f"Loading weights from file: {updated_grads_filepath}...")
        with open(updated_grads_filepath, "rb") as file:
            updated_grads_ndarrays = pickle.load(file)
        log(INFO, "Done")

        log(INFO, "Removing gradients file... ")
        os.remove(updated_grads_filepath)
        log(INFO, "Done")

        return updated_grads_ndarrays, 1, {}

    def evaluate(self, parameters, config):
        # self.model.set_weights(parameters)
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return 0.0, 1, {"custom_metric": 123}


class AsyncLocalParameterServerClient(NumPyClient):
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        glb_sig_pipe = "glb_sig"

        if config["should_stop"]:
            log(INFO, "Sending signal...")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("STOP")
                log(INFO, "Done")
                return parameters, 1, {}
        elif config["last_round"]:
            log(INFO, "Sending signal... ")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("LAST")
                log(INFO, "Done")

        global_params_filepath = "params.pkl"
        with open(global_params_filepath, "wb") as file:
            pickle.dump(parameters, file)
        log(INFO, f"Global model parameters saved: {global_params_filepath}")

        with open(glb_sig_pipe, "w") as pipe:
            log(INFO, "Sending signal...")
            pipe.write("GLB_PARAMS\n")
            log(INFO, "Done")

        while True:
            grad_files = [f for f in os.listdir() if re.match(r"grads_\d+.pkl", f)]
            if grad_files:
                break
            time.sleep(1)

        print("Got all gradient updates:", grad_files)

        updated_grads_ndarrays = []
        for f in grad_files:
            log(INFO, f"Loading weights from file: {f}...")
            with open(f, "rb") as file:
                updated_grads_ndarrays.append(pickle.load(file))
            os.remove(f)
            log(INFO, "Done")

        avg_grads_ndarrays = [
            np.mean([gradients[i] for gradients in updated_grads_ndarrays], axis=0)
            for i in range(len(updated_grads_ndarrays[0]))
        ]

        return avg_grads_ndarrays, 1, {}

    def evaluate(self, parameters, config):
        # self.model.set_weights(parameters)
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return 0.0, 1, {"custom_metric": 123}


def client_fn(context: Context):
    return AsyncLocalParameterServerClient().to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
