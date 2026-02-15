import torch
import torch.nn as nn
import torch.nn.functional as F

class TriageNet(nn.Module):
    """Enhanced Triage Network with attention mechanism for explainability"""
    def __init__(self, input_dim, num_classes):
        super(TriageNet, self).__init__()
        
        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention mechanism for feature importance
        self.attention = nn.Linear(64, 64)
        
        # Classification head
        self.fc3 = nn.Linear(64, num_classes)
        
        # Store attention weights for explainability
        self.attention_weights = None
    
    def forward(self, x):
        # First layer with batch norm and dropout
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        
        # Second layer
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        
        # Attention mechanism
        attention_scores = torch.softmax(self.attention(out), dim=1)
        self.attention_weights = attention_scores  # Store for explainability
        
        # Apply attention
        out = out * attention_scores
        
        # Final classification
        out = self.fc3(out)
        
        return out
    
    def get_attention_weights(self):
        """Return the last computed attention weights for explainability"""
        return self.attention_weights


class SimpleTriageNet(nn.Module):
    """Simpler version for backward compatibility"""
    def __init__(self, input_dim, num_classes):
        super(SimpleTriageNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)