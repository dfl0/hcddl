from flwr.client import ClientApp
from flwr.common import Context

from hcddl.local_ps_client import LocalParameterServerClient
from hcddl.async_local_ps_client import AsyncLocalParameterServerClient

from hcddl.worker_client import WorkerClient

from hcddl.task import load_data, load_model


def client_fn(context: Context):
    arch = context.run_config["arch"]
    server_type = context.run_config["server-type"]
    aggr_type = context.run_config["aggr-type"]

    print("*"*50)
    print(arch)
    print(server_type)
    print(aggr_type)
    print("*"*50)

    if server_type == "global" and arch == "hierarchical":
        return (
            LocalParameterServerClient() if aggr_type == "sync" else
            AsyncLocalParameterServerClient()
        ).to_client()

    else:
        model = load_model()

        epochs = 1
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose")

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions, batch_size)

        return WorkerClient(
            model, data, epochs, verbose
        ).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
