import torch.nn as nn
import numpy as np
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).view(1, n_position, d_hid)

    def forward(self):
        return self.pos_table.clone().detach()


class Transformer(nn.Module):
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(1, configs.hidden_dim)
        self.position_layer = PositionalEncoding(configs.hidden_dim, configs.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=configs.hidden_dim, nhead=configs.hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.layer)
        self.fc_out = nn.Linear(configs.hidden_dim * configs.seq_len,  configs.pred_len)
        self.u = configs.enc_in

    def forward(self, x):
        B, L, _ = x.shape
        x = x[:, :, :self.u]

        # Normalization
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - means) / stdev
        x = x.reshape(B*self.u, L, 1)
        x = self.embedding(x) + self.position_layer()
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        out = self.transformer_encoder(x)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        out = out.reshape(B*self.u, -1)  # Flatten to (batch_size, seq_len * hidden_dim)

        out = self.fc_out(out)
        out = out.reshape(B, -1, self.u)
        out = out * stdev + means
        return out