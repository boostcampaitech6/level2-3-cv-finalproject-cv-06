import torch
import torch.nn as nn

"""
LSTM output
- N : number of batches
- L : sequence lengh
- Q : input dim
- K : number of layers
- D : LSTM feature dimension

Y,(hn,cn) = LSTM(X)

- X : [N x L x Q] - `N` input sequnce of length `L` with `Q` dim.
- Y : [N x L x D] - `N` output sequnce of length `L` with `D` feature dim.
- hn : [K x N x D] - `K` (per each layer) of `N` final hidden state with  `D` feature dim.
- cn : [K x N x D] - `K` (per each layer) of `N` final hidden state with  `D` cell dim.
"""


class Encoder(nn.Module):
    """
    input: input_seq: (batch_size, seq_len, n_features) -> (1, 20, 38)
    output: hidden_cell -> (hn, cn)
        -> ((num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size))
    """

    def __init__(self, num_layers, hidden_size, n_features, device):
        super(Encoder, self).__init__()

        self.input_size = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def initHidden(self, batch_size):
        """
        intialize hn, cn
        """
        self.hidden_cell = (
            torch.randn(
                (self.num_layers, batch_size, self.hidden_size), dtype=torch.float
            ).to(self.device),
            torch.randn(
                (self.num_layers, batch_size, self.hidden_size), dtype=torch.float
            ).to(self.device),
        )

    def forward(self, input_seq):
        self.initHidden(input_seq.shape[0])
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        return self.hidden_cell


class Decoder(nn.Module):
    """
    input: (input_seq, hidden_cell)
        input_seq:
        hidden_cell: encoder 에서 넘어온 hidden_cell (hn, cn)
    output:
        decoder output: (batch_size, seq_len, n_features) -> (1, 1, 38)
        linear output: (batch_size, n_features) -> (1, 38)
    """

    def __init__(self, num_layers, hidden_size, n_features, device):
        super(Decoder, self).__init__()

        self.input_size = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_features)

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        output = self.linear(output)
        return output, hidden_cell


class LSTMAutoEncoder(nn.Module):
    """
    output: input seq_len(20) 모두 복원
        reconstruction 순서는 입력의 반대.
    """

    def __init__(self, num_layers, hidden_size, n_features, device):
        super(LSTMAutoEncoder, self).__init__()
        self.device = device
        self.encoder = Encoder(num_layers, hidden_size, n_features, device)
        self.decoder = Decoder(num_layers, hidden_size, n_features, device)

    def forward(self, input_seq):
        output = torch.zeros(size=input_seq.shape, dtype=torch.float)
        hidden_cell = self.encoder(input_seq)
        input_decoder = torch.zeros(
            (input_seq.shape[0], 1, input_seq.shape[2]), dtype=torch.float
        ).to(self.device)
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]

        return output.to(self.device)
