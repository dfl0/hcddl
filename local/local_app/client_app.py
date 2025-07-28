import random
import time
import numpy as np
import tensorflow as tf
from logging import INFO

from flwr.common import Context
from flwr.common.logger import log

from flwr.client import NumPyClient, ClientApp

from .task import load_data, load_optimizer, load_loss_fn, load_compiled_model


class WorkerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        # Initialize dataset that will be reused
        log(INFO, "Preprocessing dataset...")
        (x_train, y_train), (self.x_test, self.y_test) = data
        batched_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(batch_size)
        self.data_batches = list(batched_dataset)
        log(INFO, "Dataset shuffled and split up into {len(batched_dataset)} batches ({batch_size} samples each)")

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.optimizer = load_optimizer()
        self.loss_fn = load_loss_fn()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        num_samples = 0

        # Train on just one batch of the dataset
        x_train_batch, y_train_batch = random.choice(self.data_batches)

        log(INFO, "Beginning client fit step on random batch")

        # delay = int(random.random() * 20)
        # log(INFO, f"Sleeping for {delay}s...")
        # time.sleep(delay)
        # log(INFO, "Done")

        # log(INFO, f"Batch labels: {y_train_batch}")

        with tf.GradientTape() as tape:
            logits = self.model(x_train_batch, training=True)
            loss_value = self.loss_fn(y_train_batch, logits)
            if self.model.losses:
                loss_value += tf.reduce_sum(self.model.losses)

        gradients = tape.gradient(loss_value, self.model.trainable_weights)

        client_gradients = [
            grad.numpy() if grad is not None else np.zeros_like(w.numpy(), dtype=np.float32)
            for grad, w in zip(gradients, self.model.trainable_weights)
        ]

        num_samples += x_train_batch.shape[0]
        metrics = {"loss": float(loss_value)}

        log(INFO, f"Client fit: Loss = {float(loss_value):.4f}, Samples = {num_samples}")

        return client_gradients, num_samples, metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.ds_test) * self.batch_size, {"accuracy": accuracy}


def client_fn(context: Context):
    model = load_compiled_model()

    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    return WorkerClient(
        model, data, epochs, batch_size, verbose
    ).to_client()


app = ClientApp(client_fn=client_fn)
