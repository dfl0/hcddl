import numpy as np
import tensorflow as tf
import keras
from logging import INFO

from flwr.common import Context
from flwr.common.logger import log

from flwr.client import NumPyClient, ClientApp

from .task import load_data, load_compiled_model


class WorkerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        (x_train, y_train), (self.x_test, self.y_test) = data
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.optimizer = keras.optimizers.SGD(learning_rate=0.01)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        num_samples = 0

        # Train on just one batch of the dataset.
        (x_train_batch, y_train_batch) = next(iter(self.train_dataset.shuffle(1024).batch(self.batch_size if self.batch_size is not None else len(self.train_dataset))))
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

        log(INFO, f"Loss = {float(loss_value):.4f}")

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
