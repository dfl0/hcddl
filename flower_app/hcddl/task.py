import os

import keras
from keras import layers
import tensorflow as tf
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from logging import INFO
from flwr.common.logger import log


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


model = None


def load_model():
    global model
    if model is None:
        # model = keras.Sequential(
        #     [
        #         keras.Input(shape=(32, 32, 1)),
        #         layers.Conv2D(32, kernel_size=(2, 2), activation="relu"),
        #         layers.MaxPooling2D(pool_size=(2, 2)),
        #         layers.Conv2D(16, kernel_size=(2, 2), activation="relu"),
        #         layers.MaxPooling2D(pool_size=(2, 2)),
        #         layers.Flatten(),
        #         layers.Dropout(0.2),
        #         layers.Dense(10),
        #     ]
        # )

        # LeNet
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(6, kernel_size=(5, 5), activation="tanh"),
                layers.AveragePooling2D(pool_size=(2, 2), strides=2),
                layers.Conv2D(16, kernel_size=(5, 5), activation="tanh"),
                layers.AveragePooling2D(pool_size=(2, 2), strides=2),
                layers.Flatten(),
                layers.Dense(120, activation="tanh"),
                layers.Dense(84, activation="relu"),
                layers.Dense(10),
            ]
        )

        # base_model = keras.applications.MobileNetV3Small(
        #     input_shape=(32, 32, 1),
        #     include_top=False,
        #     weights=None,
        #     pooling="avg",
        # )

        # model = keras.Sequential([
        #     base_model,
        #     keras.layers.Dense(10),
        # ])

    return model


optimizer = None
loss_fn = None


def load_optimizer():
    global optimizer
    if optimizer is None:
        # optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        optimizer = keras.optimizers.Adam()
    return optimizer


def load_loss_fn():
    global loss_fn
    if loss_fn is None:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn


def load_compiled_model():
    model = load_model()
    model.compile(
        optimizer=load_optimizer(),
        loss=load_loss_fn(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


# The following are the 80/20 split of only the 'train' split of cifar10 (50k examples)
batched_train_ds = None
# batched_test_ds = None


def load_data(partition_id, num_partitions, batch_size):
    global batched_train_ds, batched_test_ds

    # if batched_train_ds is None or batched_test_ds is None:
    if batched_train_ds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            # dataset="Mike0307/MNIST-M",
            # dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

        log(INFO, "Preprocessing dataset...")

        partition = fds.load_partition(partition_id, "train")
        partition.set_format("numpy")

        partition = partition.select(range(2000))  # load only a portion of the samples

        # partition = partition.train_test_split(test_size=0.2)  # Divide data on each node: 80% train, 20% test
        # x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["label"]
        # x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["label"]

        x_train, y_train = partition["image"] / 255.0, partition["label"]

        # x_train = x_train[..., tf.newaxis]
        # x_train = tf.image.resize(x_train, [32, 32]).numpy()

        batched_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        # batched_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        log(INFO, f"Dataset split up into {len(batched_train_ds)} batches ({batch_size} samples each, {x_train.shape[0]} samples total)")

    return batched_train_ds  # , batched_test_ds


test_ds = None


def load_test_data():
    global test_ds
    if test_ds is None:
        test_ds = load_dataset("zalando-datasets/fashion_mnist", split="test").with_format("numpy")
        # test_ds = load_dataset("uoft-cs/cifar10", split="test").with_format("numpy")

        x_test, y_test = test_ds["image"] / 255.0, test_ds["label"]
        # x_test = x_test[..., tf.newaxis]
        # x_test = tf.image.resize(x_test, [32, 32]).numpy()

        test_ds = (x_test, y_test)

    return test_ds
