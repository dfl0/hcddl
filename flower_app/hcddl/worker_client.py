import timeit
import numpy as np
import tensorflow as tf

from logging import INFO

from flwr.client import NumPyClient
from flwr.common.logger import log

from hcddl.task import load_optimizer, load_loss_fn


train_ds_it = None


class WorkerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, verbose
    ):
        self.data = data

        global train_ds_it
        if train_ds_it is None:
            train_ds_it = iter(self.data)

        self.model = model
        self.epochs = epochs
        self.verbose = verbose

        self.optimizer = load_optimizer()
        self.loss_fn = load_loss_fn()

        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_loss_metric = tf.keras.metrics.Mean()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        global train_ds_it
        try:
            data_batch = next(train_ds_it)
        except StopIteration:
            train_ds_it = iter(self.data)
            data_batch = next(train_ds_it)

        x_train, y_train = data_batch

        log(INFO, "Beginning client fit step on next batch")
        log(INFO, f"Batch labels: {y_train}")

        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss_value = self.loss_fn(y_batch, logits)
                if self.model.losses:
                    loss_value += tf.reduce_sum(self.model.losses)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights))

            self.train_acc_metric.update_state(y_batch, logits)
            self.train_loss_metric.update_state(loss_value)

            return grads

        # keep track of time taken for computation in the fit step
        start_time = timeit.default_timer()
        gradients = train_step(x_train, y_train)
        train_step(x_train, y_train)
        elapsed = timeit.default_timer() - start_time
        log(INFO, "Worker step time: %.4f", elapsed)

        client_gradients = [
            grad.numpy() if grad is not None else np.zeros_like(w.numpy(), dtype=np.float32)
            for grad, w in zip(gradients, self.model.trainable_weights)
        ]

        num_samples = x_train.shape[0]

        metrics = {
            "loss": float(self.train_loss_metric.result().numpy()),
            "acc": float(self.train_acc_metric.result().numpy()),
            "comp_time": elapsed,
        }

        log(INFO, f"Client fit: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['acc']:.4f}, Samples = {num_samples}")

        self.train_acc_metric.reset_state()
        self.train_loss_metric.reset_state()

        return client_gradients, num_samples, metrics

    # def evaluate(self, parameters, config):
    #     self.model.set_weights(parameters)
    #     loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
    #     return loss, len(self.ds_test) * self.batch_size, {"accuracy": accuracy}
