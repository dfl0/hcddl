import os
import time
import pickle

from logging import INFO

from flwr.client import NumPyClient
from flwr.common.logger import log


class LocalParameterServerClient(NumPyClient):
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        glb_sig_pipe = "glb_sig"

        if config["should_stop"]:
            log(INFO, "")
            log(INFO, "Sending signal...")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("STOP\n")  # -> Local PS
            return parameters, 1, {}
        elif config["last_round"]:
            log(INFO, "")
            log(INFO, "Sending signal...")
            with open(glb_sig_pipe, "w") as pipe:
                pipe.write("LAST\n")  # -> Local PS
            time.sleep(1)

        global_params_filepath = "params.pkl"
        with open(global_params_filepath, "wb") as file:
            pickle.dump(parameters, file)
        log(INFO, f"Global model params saved: {global_params_filepath}")

        log(INFO, "")
        log(INFO, "Sending signal...")
        glb_sig_pipe = "glb_sig"
        with open(glb_sig_pipe, "w") as pipe:
            pipe.write("GLB_AGGR_W\n")  # -> Local PS

        log(INFO, "Waiting for local training and aggregation to finish...")

        loc_sig_pipe = "loc_sig"
        if not os.path.exists(loc_sig_pipe):
            os.mkfifo(loc_sig_pipe)

        with open(loc_sig_pipe, "r") as pipe:
            signal = pipe.readline().strip()  # wait for LOC_GRAD_W from Local PS
        log(INFO, f"Signal received: {signal}")
        log(INFO, "")
        os.remove(loc_sig_pipe)

        updated_grads_filepath = "grads.pkl"
        log(INFO, f"Loading weights from file: {updated_grads_filepath}...")
        with open(updated_grads_filepath, "rb") as file:
            updated_grads_ndarrays = pickle.load(file)

        log(INFO, "Removing gradients file... ")
        os.remove(updated_grads_filepath)

        return updated_grads_ndarrays, 1, {}

    # def evaluate(self, parameters, config):
    #     self.model.set_weights(parameters)
    #     loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    #     return 0.0, 1, {}
