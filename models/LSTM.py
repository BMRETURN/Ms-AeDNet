import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.embedding = nn.Linear(1, configs.hidden_dim)
        self.lstm = nn.LSTM(input_size=configs.hidden_dim, hidden_size=configs.hidden_dim, num_layers=configs.layer, batch_first=True)
        self.fc_out = nn.Linear(configs.hidden_dim, configs.pred_len)
        self.u = configs.enc_in

    def forward(self, x):
        B, L, _ = x.shape
        x = x[:, :, :self.u]
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - means) / stdev
        x = x.reshape(B * self.u, L, 1)
        x = self.embedding(x)
        output, h_n = self.lstm(x)
        output = self.fc_out(output[:, -1, :])
        output = output.reshape(B, -1, self.u)
        output = output * stdev + means
        return output