import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

# 1. The Model Architecture
class TriageNet(nn.Module):
    def __init__(self):
        super(TriageNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4) 
        )

    def forward(self, x):
        return self.fc(x)

# 2. Extract Weights (Helper for Flower)
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# 3. Load Weights (Helper for Flower)
def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# 4. Local Training Logic
def train(net, trainloader, epochs):
    device = torch.device("cpu") # Force CPU to avoid CUDA initialization crashes
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Use set_to_none=True to fully clear memory
            optimizer.zero_grad(set_to_none=True) 
            
            # 2. Separate the forward pass for clarity and memory management
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # 3. Backward pass
            loss.backward()
            optimizer.step()

# 5. Local Evaluation Logic
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# 6. Data Ingestion for your Hospital Shards
def load_local_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Assume 'triage_level' is the label, everything else is a feature
    X = df.drop('triage_level', axis=1).values
    y = df['triage_level'].values
    
    # Standard SDE practice: Scale your features for Neural Nets
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train/test for local validation
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return DataLoader(train_ds, batch_size=32), DataLoader(test_ds, batch_size=32)