# client.py
import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import load_data
from task import TriageNet

DEVICE = torch.device("cpu")

train_x, train_y, val_x, val_y = load_data()

train_loader = DataLoader(
    TensorDataset(train_x, train_y), batch_size=16, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(val_x, val_y), batch_size=16
)

model = TriageNet(train_x.shape[1], len(set(train_y.tolist()))).to(DEVICE)

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        # set model parameters
        params_dict = dict(zip(model.state_dict().keys(), map(torch.tensor, parameters)))
        model.load_state_dict(params_dict, strict=True)

        # train locally
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for data, label in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

        return [val.cpu().numpy() for _, val in model.state_dict().items()], len(train_loader), {}

    def evaluate(self, parameters, config):
        # set parameters
        params_dict = dict(zip(model.state_dict().keys(), map(torch.tensor, parameters)))
        model.load_state_dict(params_dict, strict=True)

        model.eval()
        correct = total = 0
        loss = 0.0

        with torch.no_grad():
            for data, label in val_loader:
                out = model(data)
                loss += F.cross_entropy(out, label).item()
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        return float(loss / total), total, {"accuracy": correct / total}

# Run client
fl.client.start_client(
    server_address="127.0.0.1:8080", 
    client=FLClient().to_client()
)
