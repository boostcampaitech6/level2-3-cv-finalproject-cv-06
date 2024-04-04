import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, n_features, prediction_time):
        super(LSTMAutoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_time = prediction_time

        # Encoder
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.encoder2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)

        # Repeat vector for prediction_time
        self.repeat_vector = nn.Sequential(
            nn.ReplicationPad1d(padding=(0, prediction_time - 1)),
            nn.ReplicationPad1d(padding=(0, 0)),  # Adjusted padding
        )

        # Decoder
        self.decoder = nn.LSTM(input_size=50, hidden_size=100, batch_first=True)
        self.decoder2 = nn.LSTM(
            input_size=100, hidden_size=n_features, batch_first=True
        )

    def forward(self, x):
        # Encoder
        _, (x, _) = self.encoder(x)
        _, (x, _) = self.encoder2(x)

        # Repeat vector for prediction_time
        x = self.repeat_vector(x)

        # Decoder
        _, (x, _) = self.decoder(x)
        _, (x, _) = self.decoder2(x)

        return x


# Instantiate the model
sequence_length = 20  # Adjust as needed
prediction_time = 1  # Adjust as needed
n_features = 38  # Number of features to predict

# Create a sample input tensor
x2 = torch.rand((1, sequence_length, n_features))

model = LSTMAutoencoder(sequence_length, n_features, prediction_time)
output = model(x2)

import h5py
import torch
import torch.nn as nn

# Instantiate the PyTorch model
sequence_length = 20  # Adjust as needed
prediction_time = 1  # Adjust as needed
n_features = 38  # Number of features to predict
pytorch_model = LSTMAutoencoder(sequence_length, n_features, prediction_time)

# Load weights from Keras h5 file
keras_weights_file = "model.h5"
keras_weights = {}


def extract_weights(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        keras_weights[name] = torch.tensor(obj[()])


with h5py.File(keras_weights_file, "r") as hf:
    hf.visititems(extract_weights)

# Set PyTorch model weights
state_dict = pytorch_model.state_dict()
for name, param in state_dict.items():
    if name in keras_weights:
        print(keras_weights[name])
        param.data.copy_(keras_weights[name])

# Save PyTorch model
torch.save(pytorch_model.state_dict(), "pytorch_model.pth")
