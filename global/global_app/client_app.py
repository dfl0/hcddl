import os
import pickle
from logging import INFO

from flwr.common.logger import log
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context


# Define Flower Client and client_fn
class LocalParameterServerClient(NumPyClient):
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        glb_aggr_pipe = "glb_aggr_sig"

        if config["should_stop"]:
            log(INFO, "Sending signal...")
            with open(glb_aggr_pipe, "w") as pipe:
                pipe.write("STOP")  # -> Edge Aggregator server
                log(INFO, "Done")
                return parameters, 1, {}
        elif config["last_round"]:
            log(INFO, "Sending signal... ")
            with open(glb_aggr_pipe, "w") as pipe:
                pipe.write("LAST")  # -> Edge Aggregator server
                log(INFO, "Done")
                # return parameters, 1, {}

        global_weights_filepath = "weights.pkl"
        with open(global_weights_filepath, "wb") as file:
            pickle.dump(parameters, file)
        log(INFO, f"Global model weights saved: {global_weights_filepath}")

        log(INFO, "Sending signal...")
        glb_aggr_pipe = "glb_aggr_sig"
        with open(glb_aggr_pipe, "w") as pipe:
            pipe.write("GLB_AGGR_W")  # -> Edge Aggregator server
        log(INFO, "Done")

        log(INFO, "Waiting for local training and aggregation to finish...")

        loc_aggr_pipe = "loc_aggr_sig"
        if not os.path.exists(loc_aggr_pipe):
            os.mkfifo(loc_aggr_pipe)

        with open(loc_aggr_pipe, "r") as pipe:
            signal = pipe.readline().strip()  # wait for LOC_GRAD_W from Edge Aggregator server
        log(INFO, f"Signal received: {signal}")
        os.remove(loc_aggr_pipe)

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


def client_fn(context: Context):
    return LocalParameterServerClient().to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
