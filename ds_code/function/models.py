from torch import nn, Tensor
import torch

class CustomGRU(nn.Module):
    def __init__(self, input_size, output_size, seq_len=4, label_scaler=None):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.label_scaler = label_scaler
        
        # fully connected layers to generate initial hidden state for GRU layers
        self.init_nn = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # GRU layers
        self.flatten = nn.Flatten(1, -1)
        self.normalize = nn.LayerNorm(input_size * seq_len)
        self.gru1 = nn.GRU(input_size, 256, batch_first=True)
        self.gru2 = nn.GRU(256, 128, batch_first=True)
        self.gru3 = nn.GRU(128, 64, batch_first=True)
        
        # Final fully connected layer
        self.linear = nn.Linear(64, output_size)

    def forward(self, inp, rescale=False):
        X, init_data = inp
        X = self.flatten(X)
        X = self.normalize(X).reshape((-1, self.seq_len, self.input_size))
        init_data = self.init_nn(init_data.unsqueeze(0))
        X, _ = self.gru1(X, init_data)
        X, _ = self.gru2(X)
        X, _ = self.gru3(X)
        X = self.linear(X)
        # Rescale if needed with a standard scaler (for actual prediction)
        if rescale:
            X = self.label_scaler.inverse_transform(X)
        return X
    
    def predict(self, inp, numpy_output=True):
        X, init_data = inp
        inp = (Tensor(X), Tensor(init_data))
        self.eval()
        with torch.no_grad():
            output = self(inp, rescale=True)
        if numpy_output:
            output = output.numpy()
        return output[:, -1]
    
class StandardScaler(torch.nn.Module):
    """Scaler for normalizing and revert data to original."""
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True)

    def forward(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean