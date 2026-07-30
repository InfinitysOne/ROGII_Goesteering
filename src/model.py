import torch
import torch.nn as nn

class GeosteeringHybridModel(nn.Module):
    def __init__(self, num_features=2, window_size=50):
        super(GeosteeringHybridModel, self).__init__()

        # --------------------------------------------
        # 1. CNN Block (Feature Extraction)
        # Input Shape: (Batch, Channel, Sequence_Length)

        self.cnn = nn.Sequential(
            # Layer 1: Finds patterns in GR and Z-Signal
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2), # slicing into half of the sequence

            # Layer 2: combines easy patterns to a more complex geology feature
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # --------------------------------------------
        # 2. LSTM Block (Time Series Prediction, Sequence Learning)
        # --------------------------------------------
        # We use a bidirectional LSTM, that reads the Sequence forward and backward
        self.lstm = nn.LSTM(
            input_size=64, # output channels of the last CNN layer
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # --------------------------------------------
        # 3. Fully Connected Head (Regression for TVT)
        # --------------------------------------------
        # Since we have bidirectional=True, the output of the LSTM is doubled (128 * 2 = 256)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.GELU(),
            nn.Dropout(p=0.5), # Dropout to reduce overfitting
            nn.Linear(in_features=128, out_features=1), # Output : prediction of TVT
        )

    def forward(self, x):
        """
        The forward pass of the model.
        x has the dimension (Batch_Size, Features, Window_Size) -> e.g. (32, 2, 50)
        """

        # 1. Pass the input through the CNN
        # Output x_cnn: (Batch_Size, 32_Channels, Reduce_Length)
        x_cnn = self.cnn(x)

        # 2. Preparation for LSTM:
        # The LSTM has the dimension: (Batch_Size, Sequence_Length, Features)
        # We need to transpose the Channel- and Sequence-Dimension
        x_lstm_in = x_cnn.permute(0, 2, 1)

        # 3. Pass the CNN output through the LSTM
        # lstm_out has all timesteps and the hidden states of the LSTM
        lstm_out, (hidden, cell) = self.lstm(x_lstm_in)

        # We only need the last timestep of the LSTM output
        # hidden[-2] is the forward direction's final state (seen the whole
        # window left-to-right); hidden[-1] is the backward direction's final
        # state (seen the whole window right-to-left). Concatenating these two
        # gives full context from both directions.
        last_time_step = torch.cat([hidden[-2], hidden[-1]], dim=1) # Shape: (Batch_Size, 256)

        # 4. Pass through the final linear layers
        tvt_prediction = self.fc(last_time_step) # Shape: (Batch_Size, 1)

        return tvt_prediction.squeeze(-1)

if __name__ == "__main__":
    dummy_input = torch.randn(32, 2, 100)
    model = GeosteeringHybridModel(num_features=2, window_size=100)
    predictions = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {predictions.shape}")
    print(f"Predictions in batch: {predictions[:5]}") # shows the first 5 TVT-Values
