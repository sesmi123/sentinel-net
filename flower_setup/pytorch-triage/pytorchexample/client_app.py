import torch
import flwr as fl
from collections import OrderedDict
from pytorchexample.task import TriageNet, train, test, load_local_data

# 1. Access the local data for this specific hospital
# We pass the partition_id to load hospital_0.csv, hospital_1.csv, etc.
def get_client_data(partition_id):
    trainloader, testloader = load_local_data(f"hospital_{partition_id}.csv")
    return trainloader, testloader

# 2. Define the Flower Client
class AegisClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        """Extracts model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Updates local model weights with global weights from the server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """The 'Training' phase: Train on hospital data and return updates."""
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """The 'Validation' phase: Check accuracy against local unseen patients."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

# 3. Entry point for the Flower Simulation or Deployment
def client_fn(context):
    # Retrieve the ID for this hospital (provided by Flower)
    partition_id = context.node_config["partition-id"]
    model = TriageNet()
    trainloader, testloader = get_client_data(partition_id)
    return AegisClient(model, trainloader, testloader).to_client()

# Flower App instance
app = fl.client.ClientApp(client_fn=client_fn)