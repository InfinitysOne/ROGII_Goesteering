import torch
import torch.nn as nn

class NaiveLinearBaseline(nn.Module):
    """
    A simple Linear Regression baseline.
    Flattens the (Batch, Channels, Window_Size) tensor and predicts TVT.
    """
    def __init__(self, num_features=2, window_size=50):
        super(NaiveLinearBaseline, self).__init__()
        in_dim = num_features * window_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_dim, out_features=1)
        )

    def forward(self, x):
        # x shape: (Batch, Features, Window_Size) -> e.g., (128, 2, 50)
        return self.net(x).squeeze(-1)

if __name__ == "__main__":
    dummy_input = torch.randn(32, 2, 50)
    model = NaiveLinearBaseline(num_features=2, window_size=50)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")