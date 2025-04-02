from .modified_Linear import Modified_Linear
import torch
__all__ = ['DeepSpeech']


class FullyConnected(torch.nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
    """

    def __init__(self, n_feature: int, n_hidden: int, dropout: float,
        relu_max_clip: int=20) ->None:
        super(FullyConnected, self).__init__()
        self.fc = Modified_Linear(in_features=n_feature, out_features=n_hidden)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x=x)
        x = torch.nn.functional.hardtanh(x=x, min=0, max=self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x=x, p=self.dropout, training=
                self.training)
        return x


class DeepSpeech(torch.nn.Module):
    """DeepSpeech architecture introduced in
    *Deep Speech: Scaling up end-to-end speech recognition* :cite:`hannun2014deep`.

    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.
        n_class: Number of output classes
    """

    def __init__(self, n_feature: int, n_hidden: int=2048, n_class: int=40,
        dropout: float=0.0) ->None:
        super(DeepSpeech, self).__init__()
        self.n_hidden = n_hidden
        self.fc1 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc2 = FullyConnected(n_hidden, n_hidden, dropout)
        self.fc3 = FullyConnected(n_hidden, n_hidden, dropout)
        self.bi_rnn = torch.nn.RNN(input_size=n_hidden, hidden_size=
            n_hidden, num_layers=1, nonlinearity='relu')
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = torch.nn.Linear(in_features=n_hidden, out_features=n_class)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
        Returns:
            Tensor: Predictor tensor of dimension (batch, time, class).
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.squeeze(axis=1)
        perm1 = list(range(len(x.shape)))
        perm1[0], perm1[1] = perm1[1], perm1[0]
        x = x.transpose(perm=perm1)
        x, _ = self.bi_rnn(x)
        x = x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:]
        x = self.fc4(x)
        x = self.out(x)
        x = x.transpose(perm=[1, 0, 2])
        x = torch.nn.functional.log_softmax(x=x, dim=2)
        return x
