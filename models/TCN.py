import torch.nn as nn
import torch


class TCNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers, kernel_size=3, dilation_base=2):
        super(TCNBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation
                ),
                nn.ReLU()
            ]
            input_channels = output_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()
        self.embedding = nn.Linear(1, configs.hidden_dim)
        self.tcn = TCNBlock(
            input_channels=configs.hidden_dim,
            output_channels=configs.hidden_dim,
            num_layers=configs.layer,
            kernel_size=3,
            dilation_base=2
        )
        self.fc_out = nn.Linear(configs.hidden_dim, configs.pred_len)
        self.u = configs.enc_in

    def forward(self, x):
        B, L, _ = x.shape
        x = x[:, :, :self.u]

        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - means) / stdev

        x = x.reshape(B * self.u, L, 1)  # (B*Features, SeqLen, 1)
        x = self.embedding(x)  # (B*u, L, hidden_dim)

        x = x.permute(0, 2, 1)  # (B*u, hidden_dim, L)
        x = self.tcn(x)  # (B*u, hidden_dim, L)

        x = x[:, :, -1]  # (B*u, hidden_dim)
        output = self.fc_out(x)  # (B*u, pred_len)
        output = output.reshape(B, -1, self.u)  # (B, pred_len, u)

        output = output * stdev + means
        return output