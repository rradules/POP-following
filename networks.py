import torch.nn as nn


class Mlp(nn.Module):
    """
    A pop following MLP.
    """

    def __init__(self, d_in, d_out, dropout):
        super(Mlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, d_out),
        )

    def forward(self, x):
        return self.layers(x)


class MlpLarge(nn.Module):
    """
    A pop following MLP.
    """

    def __init__(self, d_in, d_out, dropout):
        super(MlpLarge, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.layers(x)


class MlpSmall(nn.Module):
    """
    A pop following MLP.
    """

    def __init__(self, d_in, d_out, dropout):
        super(MlpSmall, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in, 10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(10, d_out),
        )

    def forward(self, x):
        return self.layers(x)


class MlpWithBatchNorm(nn.Module):
    """
    A pop following MLP with batch normalisation.
    """

    def __init__(self, d_in, d_out, dropout):
        super(MlpWithBatchNorm, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, d_out),
        )

    def forward(self, x):
        return self.layers(x)
