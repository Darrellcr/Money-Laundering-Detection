import torch
import torch.nn as nn
import platform

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

def check_memory_leak():
    input_size = 256
    hidden_size = 256
    batch_size = 16
    sequence_length = 10
    num_iterations = 100000  # Set a high number to check for memory leaks

    # Use MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Model initialization
    model = LSTMModel(input_size, hidden_size).to(device)

    # Input data and hidden state initialization
    x = torch.randn(batch_size, sequence_length, input_size).to(device)
    hidden = (
        torch.zeros(1, batch_size, hidden_size).to(device),
        torch.zeros(1, batch_size, hidden_size).to(device),
    )

    print("Starting memory check...")
    for i in range(num_iterations):
        with torch.no_grad():
            output, hidden = model(x, hidden)
        
        # Clear MPS memory cache
        torch.mps.empty_cache()
        
        print(f"Iteration {i + 1}/{num_iterations}: Completed")

if __name__ == "__main__":
    print("PyTorch Version:", torch.__version__)
    print("Python Version:", platform.python_version())
    print("Platform:", platform.system(), platform.release())
    print("MPS Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())

    check_memory_leak()