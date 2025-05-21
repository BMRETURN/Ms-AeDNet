import torch
import torch.nn as nn
import numpy as np


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

class LearnableFrequencyDecomposition(nn.Module):
    def __init__(self, initial_cutoff=0.1, temperature=10.0):
        super().__init__()
        self.cutoff = nn.Parameter(torch.tensor(initial_cutoff))
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        B, D = x.shape

        # FFT processing
        fft_coeff = torch.fft.fft(x, dim=-1)
        freqs = torch.fft.fftfreq(D).to(x.device)
        abs_freqs = freqs.abs()

        # Learnable frequency masking
        low_mask = torch.sigmoid((self.cutoff - abs_freqs) * self.temperature)
        low_mask = low_mask.unsqueeze(0)

        low_freq_coeff = fft_coeff * low_mask
        high_freq_coeff = fft_coeff * (1 - low_mask)

        trend = torch.fft.ifft(low_freq_coeff, dim=-1).real
        seasonal = torch.fft.ifft(high_freq_coeff, dim=-1).real
        return trend, seasonal

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + x
        x = self.layer_norm(x)
        return x


class MultiLayerAttention(nn.Module):
    def __init__(self, seq_len, d_model, num_layers, num_heads, dropout=0.1):
        super(MultiLayerAttention, self).__init__()
        self.position_layer = PositionalEncoding(d_model, seq_len)
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList([
            AttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x + self.position_layer()
        for i in range(self.num_layers):
            x = self.attn_layers[i](x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        """
        query: (B*NP, 5, D) -> voltage
        key_value: (B*NP, M-5, D) -> factor
        """
        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value
        )
        output = self.norm(query + self.dropout(attn_output))
        return output


class Predictor(nn.Module):
    def __init__(self, patch_num, input_dim, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.mlpdim = input_dim * (patch_num + 2)
        self.predictor = nn.Linear(self.mlpdim, pred_len)

    def forward(self, x, global_trend, global_seasonal):
        """(B*5, NP, D)"""
        B5, NP, D = x.shape
        x = x.reshape(B5, NP*D)
        final = torch.cat((global_trend, global_seasonal, x), dim=-1)
        output = self.predictor(final)  # (B*5, pred_len)
        return output.view(-1, 5, self.pred_len)  # (B, 5, pred_len)


class CausalSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.register_buffer("causal_mask", None)

    def forward(self, x):
        # x shape: (B, seq_len, D)
        if self.causal_mask is None or self.causal_mask.shape[-1] != x.shape[1]:
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool()
            self.register_buffer("causal_mask", mask.to(x.device))

        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self.causal_mask,
            need_weights=False
        )
        return attn_output


class MsAeDNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_dim = configs.patch_dim
        self.point_dim = configs.point_dim
        self.patch_head = configs.heads
        self.enc_in = configs.enc_in

        self.num_layer = nn.Linear(1, self.point_dim)
        self.attn_layer = MultiLayerAttention(self.patch_size, self.point_dim, num_layers=2, num_heads=4,
                                              dropout=0.1)
        self.patch_layer = nn.Linear(self.patch_size * self.point_dim, self.patch_dim)
        self.factor_encoder = nn.Linear(self.patch_size, self.patch_dim)

        self.cross_attn = CrossAttentionLayer(
            embed_dim=self.patch_dim,
            num_heads=self.patch_head
        )

        self.causal_attn = CausalSelfAttentionLayer(
            embed_dim=self.patch_dim,
            num_heads=4
        )

        self.predictor = Predictor(
            patch_num=self.patch_num,
            input_dim=self.patch_dim,
            pred_len=self.pred_len
        )

        self.global_freq = LearnableFrequencyDecomposition()
        self.global_trend_encoder = nn.Linear(self.seq_len, self.patch_dim)
        self.global_seasonal_encoder = nn.Linear(self.seq_len, self.patch_dim)

    def forward(self, x):
        B, L, M = x.shape
        x = x.permute(0, 2, 1)  # (B, M, L)

        # Normalization
        voltage_data = x[:, :self.enc_in, :]  # (B,5,L)
        means = voltage_data.mean(-1, keepdim=True).detach()  # (B,5,L)
        stdev = torch.sqrt(torch.var(voltage_data, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized_voltage = (voltage_data - means) / stdev

        # Global
        normalized_voltage_re = normalized_voltage.reshape(B*self.enc_in, L)
        global_trend, global_seasonal = self.global_freq(normalized_voltage_re)
        global_trend = self.global_trend_encoder(global_trend)  # (B*5,D)
        global_seasonal = self.global_seasonal_encoder(global_seasonal)  # (B*5,D)

        external_data = x[:, self.enc_in:, :]  # (B,M-5,L)
        x = torch.cat([normalized_voltage, external_data], dim=1)  # (B,M,L)

        # Patchify
        x = self.padding_patch_layer(x)  # (B, M, L + stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # (B, M, num_patches, patch_size)
        x_u = x[:, :self.enc_in]
        x_o = x[:, self.enc_in:]

        x_u = x_u.reshape(B*self.enc_in*self.patch_num, self.patch_size, 1)
        voltage_emb = self.num_layer(x_u)
        voltage_emb = self.attn_layer(voltage_emb).reshape(B*self.enc_in, self.patch_num, self.patch_size*self.point_dim)
        patch_emd = self.patch_layer(voltage_emb)
        patch_emd = patch_emd.reshape(B*self.patch_num, self.enc_in, self.patch_dim)

        # Factor
        factor_emb = self.factor_encoder(x_o)  # (B, M-5, NP, D)
        factor_emb = factor_emb.permute(0, 2, 1, 3)  # (B, NP, M-5, D)
        factor_emb = factor_emb.reshape(B * self.patch_num, M - self.enc_in, -1)  # (B*NP, M-5, D)

        # Cross Attention
        cross_output = self.cross_attn(patch_emd, factor_emb)
        cross_output = cross_output.reshape(B, self.patch_num, self.enc_in, -1)
        cross_output = cross_output.reshape(B*self.enc_in, self.patch_num, -1)

        causal_output = self.causal_attn(cross_output)

        # Prediction
        prediction = self.predictor(causal_output, global_trend, global_seasonal)  # (B, 5, pred_len)
        predictions = prediction * stdev + means

        return predictions.permute(0, 2, 1)  # (B, pred_len, 5)